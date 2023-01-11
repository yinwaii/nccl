#include "algo_interface.h"
#include <assert.h>

// Topo

ncclTopoRing2D::ncclTopoRing2D(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_BUTTERFLY_2D, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoRing2D::topoPreset(struct ncclTopoRanks *topoRanks) {
  // int rank = comm->rank;
  // int localRanks = comm->localRanks;
  // int nChannels = comm->nChannels;

  // intraRanks = new int[localRanks * MAXCHANNELS];

  // for (int c = 0; c < nChannels; c++) {
  //   struct ncclChannel *channel = comm->channels + c;
  //   int *butterfly2DIntra = graph.intra + c * localRanks;
  //   for (int i = 0; i < localRanks; i++) {
  //     if (butterfly2DIntra[i] == rank) {
  //       topoRanks->internalRank[c] = i;
  //       channel->butterfly2d.intra_prev = (i == 0) ? butterfly2DIntra[localRanks - 1] : butterfly2DIntra[i - 1];
  //       channel->butterfly2d.intra_next = (i == localRanks - 1) ? butterfly2DIntra[0] : butterfly2DIntra[i + 1];
  //     }
  //   }
  //   for (int i = 0; i < localRanks; i++) {
  //     int intraPeer = (topoRanks->internalRank[c] + i) % localRanks;
  //     intraRanks[c * localRanks + i] = butterfly2DIntra[intraPeer];
  //   }
  // }

  return ncclSuccess;
}

ncclResult_t ncclTopoRing2D::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  // int nChannels = comm->nChannels, nRanks = comm->nRanks;
  // int nNodes = comm->nNodes, localRanks = comm->localRanks;
  // int node = comm->node, rank = comm->rank;
  // if (localRanks % nNodes != 0 || nRanks != nNodes * localRanks)
  //   return ncclInvalidUsage;

  // peerRanks = new int[log2i(nNodes) * MAXCHANNELS];

  // for (int c = 0; c < nChannels; c++) {
  //   int localRank = allTopoRanks[comm->rank]->internalRank[c];
  //   struct ncclChannel *channel0 = comm->channels + c, *channel1 = channel0 + nChannels;

  //   for (int r = 0; r < nRanks; r++) {
  //     int r_node = allTopoRanks[r]->node;
  //     int r_localRank = allTopoRanks[r]->internalRank[c];
  //     if (r_localRank == localRank) {
  //       int edge = 1 << log2i(nNodes), edgePeer = node ^ edge;
  //       if (r_node == edgePeer)
  //         channel0->butterfly2d.edgeRank = channel1->butterfly2d.edgeRank = r;
  //       for (int mask = 0; mask < log2i(nNodes); mask++) {
  //         int peer = node ^ (1 << mask);
	// 				if ((node & edge == 0) && r_node == peer)
	// 					peerRanks[c * log2i(nNodes) + mask] = r;
  //       }
  //     }
  //   }
  //   // TRACE(NCCL_GRAPH, "Mesh %d-%d: %d(up) -> %d(left) -> %d(mirror) -> %d(right) -> %d(down)", comm->rank, c, channel0->Ring2D.inter_prev, channel0->Ring2D.intra_prev, channel0->Ring2D.mirror, channel0->Ring2D.intra_next, channel0->Ring2D.inter_next);
  //   // TRACE(NCCL_GRAPH, "Mesh %d-%d: %d(up) -> %d(left) -> %d(mirror) -> %d(right) -> %d(down)", comm->rank, c + nChannels, channel1->Ring2D.inter_prev, channel1->Ring2D.intra_prev, channel1->Ring2D.mirror, channel1->Ring2D.intra_next, channel1->Ring2D.inter_next);
  // }
  return ncclSuccess;
}

ncclResult_t ncclTopoRing2D::transportSetup() {
  // for (int c = 0; c < comm->nChannels; c++) {
  //   struct ncclChannel *channel = comm->channels + c;
  //   if (comm->nRanks == 1) continue;
  //   NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->Ring2D.inter_prev, 1, &channel->Ring2D.inter_next));
  //   NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->Ring2D.intra_prev, 1, &channel->Ring2D.intra_next));
  //   NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->Ring2D.mirror, 1, &channel->Ring2D.mirror));
  // }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternBroadcast;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternRing2D;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::enqueuePattern(struct ncclInfo *info, bool *redirect) const {
  if (info->coll == ncclCollBroadcast) {
    info->algorithm = NCCL_ALGO_RING;
    *redirect = true;
    return ncclSuccess;
  }
  NCCLCHECK(this->ncclEnqueueBase::enqueuePattern(info, redirect));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::proxySaveColl(struct ncclProxyArgs *args,
                                                 struct ncclInfo *info) const {
  // int pattern = info->pattern;
  // struct ncclRing2D *butterfly2D = &args->channel->Ring2D;
  // int nRanks = info->comm->nRanks;
  // if (pattern == ncclPatternRing2D) {
  //   NCCLCHECK(SaveProxy<proxyRecv>(butterfly2D->inter_prev, args));
  //   NCCLCHECK(SaveProxy<proxySend>(butterfly2D->inter_next, args));
  //   NCCLCHECK(SaveProxy<proxySend>(butterfly2D->mirror, args));
  //   NCCLCHECK(SaveProxy<proxyRecv>(butterfly2D->intra_prev, args));
  //   NCCLCHECK(SaveProxy<proxySend>(butterfly2D->intra_next, args));
  // }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternRing2D:
    info->nchunksPerLoop = 1;
    info->nstepsPerLoop = 1;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}
