#include "core.h"
#include "graph.h"

ncclResult_t ncclTopoGraphCopy(struct ncclGraphInfo* dst, struct ncclTopoGraph* src) {
	dst->pattern = src->pattern;
	dst->nChannels = src->nChannels;
	dst->sameChannels = src->sameChannels;
	dst->bwIntra = src->bwIntra;
	dst->bwInter = src->bwInter;
	dst->typeIntra = src->typeIntra;
	dst->typeInter = src->typeInter;
	return ncclSuccess;
}

ncclResult_t ncclTopoGraphFit(struct ncclTopoGraph* dst, struct ncclGraphInfo* src) {
	dst->nChannels = std::min(src->nChannels, dst->nChannels);
	dst->sameChannels = std::min(src->sameChannels, dst->sameChannels);
	dst->bwIntra = std::min(src->bwIntra, dst->bwIntra);
	dst->bwInter = std::min(src->bwInter, dst->bwInter);
	dst->typeIntra = std::max(src->typeIntra, dst->typeIntra);
	dst->typeInter = std::max(src->typeInter, dst->typeInter);
	return ncclSuccess;
}