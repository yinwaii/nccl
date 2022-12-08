#ifndef __BUTTERFLY_H__
#define __BUTTERFLY_H__
#include "base.h"
#include "comm.h"
#include "info.h"
#include "ring.h"

class ncclTopoButterfly : public ncclTopoBase {
private:
public:
  int *peerRanks;
  ncclTopoButterfly(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks,
                           struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
};

class ncclEnqueueButterfly : public ncclEnqueueBase {
private:
  enum Patterns { ncclPatternButterfly, ncclPatternHalfDoubling };
  int getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info,
                size_t size) const;

public:
  ncclEnqueueButterfly() : ncclEnqueueBase("Butterfly") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueuePattern(struct ncclInfo *info, bool *redirect) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
	ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args,
                             struct ncclInfo *info) const;
};

using ncclTuningButterfly = ncclTuningRing;

#endif