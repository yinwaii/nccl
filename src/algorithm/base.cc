#include "base.h"
#include "algo_interface.h"

// Topo

ncclTopoBase::ncclTopoBase(int id, struct ncclComm *comm, int crossNic, int collNet) : comm(comm) {
  graph.id = id;
  graph.crossNic = crossNic;
  graph.collNet = collNet;
}

ncclResult_t ncclTopoBase::graphInit(int pattern, int minChannels, int maxChannels) {
  graph.pattern = pattern;
  graph.minChannels = minChannels;
  graph.maxChannels = maxChannels;
  NCCLCHECK(ncclTopoCompute(comm->topo, &graph));
  NCCLCHECK(ncclTopoPrintGraph(comm->topo, &graph));
  return ncclSuccess;
};

ncclResult_t ncclTopoBase::graphCopy(struct ncclGraphInfo* dst) {
  dst->pattern = graph.pattern;
  dst->sameChannels = graph.sameChannels;
  dst->speedInter = graph.speedInter;
  dst->speedIntra = graph.speedIntra;
  dst->typeIntra = graph.typeIntra;
  dst->typeInter = graph.typeInter;
  return ncclSuccess;
}

ncclResult_t ncclTopoBase::graphFit(struct ncclGraphInfo* src) {
  graph.sameChannels = std::min(src->sameChannels, graph.sameChannels);
  graph.speedIntra = std::min(src->speedIntra, graph.speedIntra);
  graph.speedInter = std::min(src->speedInter, graph.speedInter);
  graph.typeIntra = std::min(src->typeIntra, graph.typeIntra);
  graph.typeInter = std::min(src->typeInter, graph.typeInter);
  return ncclSuccess;
}

// Enqueue

ncclResult_t ncclEnqueueBase::getPattern(int coll, int *pattern) const {
  *pattern = -1;
  return ncclSuccess;
}

ncclResult_t ncclEnqueueBase::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->tuning[algorithm].bandwidths[info->coll][protocol];
  float lat = info->comm->tuning[algorithm].latencies[info->coll][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
    if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclEnqueueBase::enqueuePattern(struct ncclInfo* info, bool *redirect) const {
  *redirect = false;
  NCCLCHECK(this->getPattern(info->coll, &info->pattern));
  if (info->pattern < 0) {
    WARN("Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueBase::enqueueLoopInfo(struct ncclInfo *info) const {
  WARN("Unknown pattern %d\n", info->pattern);
  return ncclInternalError;
}

ncclResult_t ncclEnqueueBase::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem* work) const {
  switch(info->protocol) {
    case NCCL_PROTO_LL: {
      const ssize_t sliceSize = sliceInfo->stepSize * sizeof(uint64_t) / sizeof(union ncclLLFifoLine);
      const ssize_t loopSize = info->nChannels * info->nchunksPerLoop * (ssize_t)sliceSize;
      work->coll.lastChunkSize = DIVUP((info->nBytes - (info->nBytes / loopSize) * loopSize), info->nChannels * info->nchunksPerLoop);
      ALIGN_SIZE(work->coll.lastChunkSize, info->nThreads * sizeof(uint64_t));
      work->coll.lastChunkSize /= ncclTypeSize(info->datatype);
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueBase::enqueueChannelThread(struct ncclInfo *info) const {
  ncclComm *comm = info->comm;
  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels; // CollNet uses one channel for up and one channel for down
  int nt = info->comm->tuning[info->algorithm].maxThreads[info->protocol];
  int threadThreshold = comm->tuning[info->algorithm].threadThresholds[info->protocol];
  while (info->nBytes < nc*nt*threadThreshold) {
    if (nc >= 2) nc--;
    else if ((nt % 128) == 0) nt/=2;
    else break;
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) nt += WARP_SIZE; // Extra warp for sync
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}

// Tuning

ncclResult_t ncclTuningBase::tuningMaxThreads(int a) {
  comm->tuning[a].maxThreads[NCCL_PROTO_SIMPLE] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->tuning[a].maxThreads[NCCL_PROTO_LL] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->tuning[a].maxThreads[NCCL_PROTO_LL128] = getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
  return ncclSuccess;
}

ncclResult_t ncclTuningBase::tuningThresholds(int a) {
  comm->tuning[a].threadThresholds[NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
  comm->tuning[a].threadThresholds[NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
  comm->tuning[a].threadThresholds[NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  return ncclSuccess;
}