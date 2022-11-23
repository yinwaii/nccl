#include "algo_interface.h"

// Topo

ncclTopoButterfly::ncclTopoButterfly(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_BUTTERFLY, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoButterfly::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank, nranks = comm->nRanks;
  int nChannels = comm->nChannels;

  NCCLCHECK(ncclCalloc(&lastRanks, nranks * MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&peerRanks, (log2i(nranks - 1) + 1) * MAXCHANNELS));

  for (int c=0; c<nChannels; c++) {
    for (int r = 0; r < nranks; r++){
      lastRanks[(c+nChannels)*nranks+r] = lastRanks[c*nranks+r] = -1;
    }
    for (int mask = 0; (1 << mask) < nranks; mask++) {
      int peer = rank ^ (1 << mask);
      peerRanks[(c+nChannels)*nranks+mask] = peerRanks[c*nranks+mask] = (peer < nranks) ? peer : -1;
    }
    topoRanks->butterflyLastRank[c] = (rank & (nranks - 1)) != rank;
    lastRanks[(c+nChannels)*nranks+0] = lastRanks[c*nranks+0] = topoRanks->butterflyLastRank[c] ? 0 : -1;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  // Gather data from all ranks
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  int lastNum = 0;
  for (int i = 0; i < nranks; i++) {
	  for (int c = 0; c < nChannels; c++) {
      // struct ncclChannel *channel0 = comm->channels + c;
      // struct ncclChannel *channel1 = channel0 + nChannels;
      if (comm->rank == 0 && allTopoRanks[i]->butterflyLastRank[c]) {
        lastRanks[c*nranks+lastNum] = i;
        lastRanks[(c+nChannels)*nranks+lastNum] = i;
        lastNum++;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly::transportSetup() {
  int nranks = comm->nRanks;
  char line[1024] = "";
  sprintf(line + strlen(line), "Butterfly for %d\n", comm->rank);
  for (int c = 0; c < comm->nChannels; c++) {
    sprintf(line + strlen(line), "Channel %d:\n", c);
    struct ncclChannel* channel = comm->channels+c;
    if (nranks == 1) continue;
    sprintf(line + strlen(line), "Peer Ranks: ");
    for (int i = 0; i < log2i(nranks - 1) + 1; i++) {
      channel->butterfly.peerRanks[i] = peerRanks[c*nranks+i];
      int peer = channel->butterfly.peerRanks[i];
      sprintf(line + strlen(line), "%d/", peer);
      if (peer != -1)
        NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &peer, 1, &peer));
    }
    sprintf(line + strlen(line), "\nLast Ranks: ");
    for (int r = 0; r < nranks; r++) {
      channel->butterfly.lastRanks[r] = lastRanks[c*nranks+r];
      int peer = channel->butterfly.lastRanks[r];
      sprintf(line + strlen(line), "%d/", peer);
      if (peer != -1) {
        if (comm->rank == 0) {
          NCCLCHECK(ncclTransportP2pConnect(comm, channel, 0, NULL, 1, &peer));
        }
        else
          NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &peer, 0, NULL));
      }
    }
    sprintf(line + strlen(line), "\n");
  }
  WARN("%s", line);
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
  int nRanks = info->comm->nRanks, rank = info->comm->rank;
  if (pattern == ncclPatternButterfly) {
    for (int i = 0; i < log2i(nRanks - 1) + 1; i++) {
      int peer = butterfly->peerRanks[i];
      if (peer != -1) {
        NCCLCHECK(SaveProxy(proxySend, peer, args));
        NCCLCHECK(SaveProxy(proxyRecv, peer, args));
      }
    }
    if ((nRanks & (-nRanks)) != nRanks) {
      for (int r = 0; r < nRanks; r++) {
        int peer = butterfly->lastRanks[r];
        if (peer != -1) {
          if (rank == 0) {
            NCCLCHECK(SaveProxy(proxySend, peer, args));
          }
          else
            NCCLCHECK(SaveProxy(proxyRecv, peer, args));
        }
      }
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