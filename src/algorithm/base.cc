#include "base.h"
#include "../graph/tuning.h"

ncclAlgoBase::ncclAlgoBase(int crossNic, int collNet) {
	graph.crossNic = crossNic;
	graph.collNet = collNet;
}

ncclResult_t ncclAlgoBase::graphInit(struct ncclComm *comm, int id, int pattern, ncclTopoSystem* system, int minChannels, int maxChannels) {
	this->comm = comm;
	graph.id = id;
	graph.pattern = pattern;
	graph.minChannels = minChannels;
	graph.maxChannels = maxChannels;
	NCCLCHECK(ncclTopoCompute(system, &graph));
	NCCLCHECK(ncclTopoPrintGraph(system, &graph));
	return ncclSuccess;
}

ncclResult_t ncclAlgoBase::graphCopy(struct ncclGraphInfo* dst) {
	dst->sameChannels = graph.sameChannels;
	dst->speedInter = graph.speedInter;
	dst->speedIntra = graph.speedIntra;
	dst->typeIntra = graph.typeIntra;
	return ncclSuccess;
}

ncclResult_t ncclAlgoBase::graphFit(struct ncclGraphInfo* src) {
	graph.sameChannels = std::min(src->sameChannels, graph.sameChannels);
	graph.speedIntra = std::min(src->speedIntra, graph.speedIntra);
	graph.speedInter = std::min(src->speedInter, graph.speedInter);
	graph.typeIntra = std::min(src->typeIntra, graph.typeIntra);
	return ncclSuccess;
}

ncclResult_t ncclAlgoBase::tuningMaxThreads(int a) {
  comm->maxThreads[a][NCCL_PROTO_SIMPLE] = comm->maxThreads[a][NCCL_PROTO_LL] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[a][NCCL_PROTO_LL128] = getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::tuningThresholds(int a) {
  comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
  comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
  comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  return ncclSuccess;
}

ncclAlgoBase::~ncclAlgoBase() {}

ncclResult_t ncclAlgoBase::getPattern(int coll, int *pattern) const {
  *pattern = -1;
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::enqueuePattern(struct ncclInfo* info) const {
  NCCLCHECK(this->getPattern(info->coll, &info->pattern));
  if (info->pattern < 0) {
    WARN("Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::enqueueLoopInfo(struct ncclInfo *info) const {
  WARN("Unknown pattern %d\n", info->pattern);
  return ncclInternalError;
}

ncclResult_t ncclAlgoBase::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl* coll) const {
  switch(info->protocol) {
    case NCCL_PROTO_LL: {
      const ssize_t sliceSize = sliceInfo->stepSize * sizeof(uint64_t) / sizeof(union ncclLLFifoLine);
      const ssize_t loopSize = info->nChannels * info->nchunksPerLoop * (ssize_t)sliceSize;
      coll->args.coll.lastChunkSize = DIVUP((info->nBytes - (info->nBytes / loopSize) * loopSize), info->nChannels * info->nchunksPerLoop);
      ALIGN_SIZE(coll->args.coll.lastChunkSize, info->nThreads * sizeof(uint64_t));
      coll->args.coll.lastChunkSize /= ncclTypeSize(info->datatype);
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::enqueueChannelThread(struct ncclInfo *info) const {
  ncclComm *comm = info->comm;
  int nc = comm->nChannels; // CollNet uses one channel for up and one channel for down
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
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