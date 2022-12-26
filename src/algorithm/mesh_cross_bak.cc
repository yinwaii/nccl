#include "algo_interface.h"
#include <assert.h>

// Topo

ncclTopoMeshCross::ncclTopoMeshCross(struct ncclComm *comm)
    : ncclTopoBase(NCCL_ALGO_MESH_CROSS, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoMeshCross::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    channel->meshCross.inter_prev = channel->meshCross.inter_next = -1;
    int *meshIntra = graph.intra + c * localRanks;
    for (int i = 0; i < localRanks; i++) {
      if (meshIntra[i] == rank) {
        topoRanks->internalRank[c] = i;
        channel->meshCross.intra_prev = (i == 0) ? meshIntra[localRanks - 1] : meshIntra[i - 1];
        channel->meshCross.intra_next = (i == localRanks - 1) ? meshIntra[0] : meshIntra[i + 1];
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoMeshCross::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  int nChannels = comm->nChannels, nRanks = comm->nRanks;
  int nNodes = comm->nNodes, localRanks = comm->localRanks;
  // if (localRanks % nNodes != 0 || nRanks != nNodes * localRanks)
  //   return ncclInvalidUsage;
  int nPartitions = localRanks / nNodes;
  for (int c = 0; c < nChannels; c++) {
    int localRank = allTopoRanks[comm->rank]->internalRank[c];
    int partition = localRank / nPartitions, subRank = localRank % nPartitions;
    struct ncclChannel *channel0 = comm->channels + c, *channel1 = channel0 + nChannels;
    for (int r = 0; r < nRanks; r++) {
      int r_node = allTopoRanks[r]->node;
      int r_localRank = allTopoRanks[r]->internalRank[c];
      int r_partition = r_localRank / nPartitions, r_subRank = r_localRank % nPartitions;
      if (comm->node == r_partition && r_node == partition && subRank == r_subRank) {
        channel0->meshCross.mirror = r;
        channel1->meshCross.mirror = r;
      }
      if (comm->node == r_node && subRank == r_subRank) {
        if ((partition + 1) % nPartitions == r_partition) {
          channel0->meshCross.inter_next = r;
          channel1->meshCross.inter_next = r;
        }
        if ((r_partition + 1) % nPartitions == partition) {
          channel0->meshCross.inter_prev = r;
          channel1->meshCross.inter_prev = r;
        }
      }
    }
    TRACE(NCCL_GRAPH, "Mesh %d-%d: %d(up) -> %d(left) -> %d(mirror) -> %d(right) -> %d(down)", comm->rank, c, channel0->meshCross.inter_prev, channel0->meshCross.intra_prev, channel0->meshCross.mirror, channel0->meshCross.intra_next, channel0->meshCross.inter_next);
    TRACE(NCCL_GRAPH, "Mesh %d-%d: %d(up) -> %d(left) -> %d(mirror) -> %d(right) -> %d(down)", comm->rank, c + nChannels, channel1->meshCross.inter_prev, channel1->meshCross.intra_prev, channel1->meshCross.mirror, channel1->meshCross.intra_next, channel1->meshCross.inter_next);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoMeshCross::transportSetup() {
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.inter_prev, 1, &channel->meshCross.inter_next));
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.intra_prev, 1, &channel->meshCross.intra_next));
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.mirror, 1, &channel->meshCross.mirror));
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternBroadcast;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternMeshCross;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::enqueuePattern(struct ncclInfo *info, bool *redirect) const {
  if (info->coll == ncclCollBroadcast) {
    info->algorithm = NCCL_ALGO_RING;
    *redirect = true;
    return ncclSuccess;
  }
  NCCLCHECK(this->ncclEnqueueBase::enqueuePattern(info, redirect));
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::proxySaveColl(struct ncclProxyArgs *args,
                                                 struct ncclInfo *info) const {
  int pattern = info->pattern;
  struct ncclMeshCross *meshCross = &args->channel->meshCross;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternMeshCross) {
    NCCLCHECK(SaveProxy<proxyRecv>(meshCross->inter_prev, args));
    NCCLCHECK(SaveProxy<proxySend>(meshCross->inter_next, args));
    NCCLCHECK(SaveProxy<proxySend>(meshCross->mirror, args));
    NCCLCHECK(SaveProxy<proxyRecv>(meshCross->intra_prev, args));
    NCCLCHECK(SaveProxy<proxySend>(meshCross->intra_next, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternMeshCross:
    info->nchunksPerLoop = 1;
    info->nstepsPerLoop = 1;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}
