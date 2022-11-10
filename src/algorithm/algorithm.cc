#include "algorithm.h"
#include "../graph/tuning.h"

ncclAlgo::ncclAlgo(struct ncclComm *comm, int crossNic, int collNet, int minChannels, int maxChannels): comm(comm)
{
	graph.crossNic = crossNic;
	graph.collNet = collNet;
	graph.minChannels = minChannels;
	graph.maxChannels = maxChannels;
}

ncclResult_t ncclAlgo::graphInit(int id, int pattern, ncclTopoSystem* system) {
	graph.id = id;
	graph.pattern = pattern;
	NCCLCHECK(ncclTopoCompute(system, &graph));
	NCCLCHECK(ncclTopoPrintGraph(system, &graph));
	return ncclSuccess;
}

ncclResult_t ncclAlgo::graphCopy(struct ncclGraphInfo* dst) {
	dst->sameChannels = graph.sameChannels;
	dst->speedInter = graph.speedInter;
	dst->speedIntra = graph.speedIntra;
	dst->typeIntra = graph.typeIntra;
	return ncclSuccess;
}

ncclResult_t ncclAlgo::graphFit(struct ncclGraphInfo* src) {
	graph.sameChannels = std::min(src->sameChannels, graph.sameChannels);
	graph.speedIntra = std::min(src->speedIntra, graph.speedIntra);
	graph.speedInter = std::min(src->speedInter, graph.speedInter);
	graph.typeIntra = std::min(src->typeIntra, graph.typeIntra);
	return ncclSuccess;
}

ncclResult_t ncclAlgo::tuningMaxThreads(int a) {
  comm->maxThreads[a][NCCL_PROTO_SIMPLE] = comm->maxThreads[a][NCCL_PROTO_LL] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[a][NCCL_PROTO_LL128] = getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
  return ncclSuccess;
}

ncclResult_t ncclAlgo::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclAlgo::tuningThresholds(int a) {
  comm->threadThresholds[a][NCCL_PROTO_LL] = NCCL_LL_THREAD_THRESHOLD;
  comm->threadThresholds[a][NCCL_PROTO_LL128] = NCCL_LL128_THREAD_THRESHOLD;
  comm->threadThresholds[a][NCCL_PROTO_SIMPLE] = NCCL_SIMPLE_THREAD_THRESHOLD;
  return ncclSuccess;
}

ncclAlgo::~ncclAlgo() {}