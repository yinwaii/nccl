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
  dst->pattern = graph.pattern;
  dst->sameChannels = graph.sameChannels;
  dst->speedInter = graph.speedInter;
	dst->speedIntra = graph.speedIntra;
	dst->typeIntra = graph.typeIntra;
  dst->typeInter = graph.typeInter;
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::graphFit(struct ncclGraphInfo* src) {
	graph.sameChannels = std::min(src->sameChannels, graph.sameChannels);
	graph.speedIntra = std::min(src->speedIntra, graph.speedIntra);
	graph.speedInter = std::min(src->speedInter, graph.speedInter);
	graph.typeIntra = std::min(src->typeIntra, graph.typeIntra);
  graph.typeInter = std::min(src->typeInter, graph.typeInter);
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::tuningMaxThreads(int a) {
  comm->maxThreads[a][NCCL_PROTO_SIMPLE] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS);
  comm->maxThreads[a][NCCL_PROTO_LL] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS);
  comm->maxThreads[a][NCCL_PROTO_LL128] = getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
  return ncclSuccess;
}

ncclResult_t ncclAlgoBase::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
    if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
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

ncclResult_t ncclAlgoBase::graphDump() {
  char *line = new char[1000];
  sprintf(line + strlen(line), "\nrank: %d\n", comm->rank);
  sprintf(line + strlen(line), "id: %d\n", graph.id);
  sprintf(line + strlen(line), "collNet: %d\n", graph.collNet);
  sprintf(line + strlen(line), "minChannels: %d\n", graph.minChannels);
  sprintf(line + strlen(line), "maxChannels: %d\n", graph.maxChannels);
  sprintf(line + strlen(line), "nChannels: %d\n", graph.nChannels);
  sprintf(line + strlen(line), "nHops: %d\n", graph.nHops);
  sprintf(line + strlen(line), "pattern: %d\n", graph.pattern);
  sprintf(line + strlen(line), "sameChannels: %d\n", graph.sameChannels);
  sprintf(line + strlen(line), "speedIntra: %lf\n", graph.speedIntra);
  sprintf(line + strlen(line), "speedInter: %lf\n", graph.speedInter);
  sprintf(line + strlen(line), "typeIntra: %d\n", graph.typeIntra);
  sprintf(line + strlen(line), "typeInter: %d\n", graph.typeInter);
  // sprintf(line + strlen(line), "intra: \n");
  // for (int c = 0; c < graph.nChannels; c++)
  // {
	// for (int g = 0; g < comm->localRanks; g++)
	// 	sprintf(line + strlen(line), "%d ", (graph.intra + c * comm->localRanks)[g]);
	// sprintf(line + strlen(line), "\n");
  // }
  // sprintf(line + strlen(line), "inter: \n");
  // for (int c = 0; c < graph.nChannels; c++)
	// sprintf(line + strlen(line), "%d %d\n", graph.inter[c * 2], graph.inter[c * 2 + 1]);
  WARN("%s\n", line);
  delete[] line;
  return ncclSuccess;
}

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

ncclResult_t ncclAlgoBase::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem* work) const {
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

ncclResult_t ncclAlgoBase::enqueueChannelThread(struct ncclInfo *info) const {
  ncclComm *comm = info->comm;
  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels; // CollNet uses one channel for up and one channel for down
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