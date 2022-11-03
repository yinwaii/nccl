#include "core.h"
#include "graph.h"

ncclResult_t ncclTopoGraphCopy(struct ncclGraphInfo* dst, struct ncclTopoGraph* src) {
	dst->sameChannels = src->sameChannels;
	dst->speedInter = src->speedInter;
	dst->speedIntra = src->speedIntra;
	dst->typeIntra = src->typeIntra;
	return ncclSuccess;
}

ncclResult_t ncclTopoGraphFit(struct ncclTopoGraph* dst, struct ncclGraphInfo* src) {
	dst->sameChannels = std::min(src->sameChannels, dst->sameChannels);
	dst->speedIntra = std::min(src->speedIntra, dst->speedIntra);
	dst->speedInter = std::min(src->speedInter, dst->speedInter);
	dst->typeIntra = std::min(src->typeIntra, dst->typeIntra);
	return ncclSuccess;
}