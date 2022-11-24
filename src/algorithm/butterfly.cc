#include "algo_interface.h"

// Topo

ncclTopoButterfly::ncclTopoButterfly(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_BUTTERFLY, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoButterfly::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank, nranks = comm->nRanks;
  int nChannels = comm->nChannels;

  peerRanks = new int[log2i(nranks) * MAXCHANNELS];

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    int edge = 1 << log2i(nranks), edgePeer = rank ^ edge;
    channel->butterfly.edgeRank = edgePeer < nranks ? edgePeer : -1;
    for (int mask = 0; mask < log2i(nranks); mask++) {
      int peer = rank ^ (1 << mask);
      peerRanks[(c+nChannels)*nranks+mask] = peerRanks[c*nranks+mask] = (rank & edge) ? -1 : peer;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {

  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly::transportSetup() {
  int nranks = comm->nRanks;
  char line[1024] = "";
  sprintf(line + strlen(line), "Butterfly for %d\n", comm->rank);
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    int edgePeer = channel->butterfly.edgeRank;
    sprintf(line + strlen(line), "Channel %d: edgeRanks %d\n", c, channel->butterfly.edgeRank);
    if (edgePeer != -1)
      NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &edgePeer, 1, &edgePeer));

    if (nranks == 1) continue;
    sprintf(line + strlen(line), "Peer Ranks: ");
    for (int i = 0; i < log2i(nranks); i++) {
      channel->butterfly.peerRanks[i] = peerRanks[c*nranks+i];
      int peer = channel->butterfly.peerRanks[i];
      sprintf(line + strlen(line), "%d/", peer);
      if (peer != -1)
        NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &peer, 1, &peer));
    }
    sprintf(line + strlen(line), "\n");
  }
  delete[] peerRanks;
  INFO(NCCL_COLL, "%s", line);
  NCCLCHECK(ncclTransportP2pSetup(comm, &graph));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly::getPattern(int coll, int *pattern) const {
  switch (coll) {
    // case ncclFuncBroadcast:
    //   *pattern = ncclPatternHalfDoubling;
    //   break;
    case ncclFuncAllReduce:
      *pattern = ncclPatternButterfly;
      break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  int pattern = info->pattern;
  struct ncclButterfly *butterfly = &args->channel->butterfly;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternButterfly) {
    int edgeRank = butterfly->edgeRank;
    if (edgeRank != -1) {
      NCCLCHECK(SaveProxy(proxySend, edgeRank, args));
      NCCLCHECK(SaveProxy(proxyRecv, edgeRank, args));
    }
    for (int i = 0; i < log2i(nRanks); i++) {
      int peer = butterfly->peerRanks[i];
      NCCLCHECK(SaveProxy(proxySend, peer, args));
      NCCLCHECK(SaveProxy(proxyRecv, peer, args));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternButterfly:
    info->nchunksPerLoop = 1;
    info->nstepsPerLoop = 1;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}