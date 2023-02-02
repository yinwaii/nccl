#include "algo_interface.h"

ncclResult_t ncclEnqueueButterfly2::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternHalfDoubling;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternButterfly;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly2::enqueueRedirect(struct ncclInfo *info) const {
  if (info->coll == ncclCollBroadcast) {
    info->comm->algoEnable[NCCL_ALGO_BUTTERFLY_YZ] = 1;
    info->comm->algoEnable[NCCL_ALGO_BUTTERFLY2] = 0;
  }
  return ncclSuccess;
}

int ncclEnqueueButterfly2::getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, size_t size) const {
  // Compute nSteps for proxies
  int stepSize = info->comm->buffSizes[info->protocol] / NCCL_STEPS;
  int chunkEffectiveSize = stepSize * args->chunkSteps;
  if (info->protocol == NCCL_PROTO_LL)
    chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128)
    chunkEffectiveSize = (chunkEffectiveSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  // if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d
  // (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels,
  // info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = 2 * (int)(DIVUP(size / 2, (((size_t)(info->nChannels)) * info->nchunksPerLoop * chunkEffectiveSize)));
  return info->nstepsPerLoop * nLoops * args->chunkSteps;
}

ncclResult_t ncclEnqueueButterfly2::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const {
  int pattern = info->pattern;
  struct ncclButterfly *butterfly = &args->channel->butterfly;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternButterfly) {
    int edgeRank = butterfly->edgeRank;
    if (edgeRank != -1) {
      NCCLCHECK(SaveProxy<proxySend>(edgeRank, args));
      NCCLCHECK(SaveProxy<proxyRecv>(edgeRank, args));
    }
    for (int i = 0; i < log2i(nRanks); i++) {
      int peer = butterfly->peerRanks[i];
      int nsteps = getNsteps(args, info, (info->nBytes >> i));
      NCCLCHECK(SaveProxy<proxySend>(peer, args, nsteps));
      NCCLCHECK(SaveProxy<proxyRecv>(peer, args, nsteps));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly2::enqueueLoopInfo(struct ncclInfo *info) const {
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
