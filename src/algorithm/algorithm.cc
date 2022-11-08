#include "algorithm.h"

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