#ifndef __BUTTERFLY_YZ_H__
#define __BUTTERFLY_YZ_H__
#include "comm.h"
#include "info.h"
#include "base.h"
#include "ring.h"

class ncclTopoButterfly_yz: public ncclTopoBase {
private:
	ncclResult_t connectButterfly(struct ncclComm *comm, int *butterflyRecv, int *butterflySend, int *firstRanks);

public:
	int *peerRanks;
	ncclTopoButterfly_yz(struct ncclComm *comm);
	ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
	ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
	ncclResult_t transportSetup();
};

class ncclEnqueueButterfly_yz: public ncclEnqueueBase {
private:
  enum Patterns {
    ncclPatternButterfly,
    ncclPatternHalfDoubling
  };

public:
  ncclEnqueueButterfly_yz(): ncclEnqueueBase("Butterfly_yz") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

using ncclTuningButterfly_yz = ncclTuningRing;

#endif