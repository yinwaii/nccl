#ifndef __TREE_H__
#define __TREE_H__
#include "comm.h"
#include "info.h"
#include "base.h"
#include "ring.h"

class ncclTopoButterfly: public ncclTopoBase {
private:
public:
  int *treePatterns;
  ncclTopoButterfly(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
};

class ncclEnqueueButterfly: public ncclEnqueueBase {
private:
  enum Patterns {
    ncclPatternButterfly,
  };

public:
  ncclEnqueueButterfly(): ncclEnqueueBase("Butterfly") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem *work) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

using ncclTuningButterfly = ncclTuningRing;

#endif