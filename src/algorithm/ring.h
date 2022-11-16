#ifndef __RING_H__
#define __RING_H__
#include "base.h"
#include "comm.h"
#include "info.h"

class ncclAlgoRing : public ncclAlgoBase
{
private:
  enum Patterns {
    ncclPatternRing,
    ncclPatternRingTwice,
    ncclPatternPipelineFrom,
    ncclPatternPipelineTo,
  };
  ncclResult_t connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks);
  void dumpLine(int *values, int nranks, const char *prefix);
  ncclResult_t ncclBuildRings(int nrings, int *rings, int rank, int nranks, int *prev, int *next);
  ncclResult_t setupChannel(int channelId, int rank, int nranks, int *ringRanks);
  bool NeedProxy(int type, int pattern, int root, struct ncclRing *ring, int nranks) const;

public:
  int *rings;
  ncclAlgoRing();
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  ncclResult_t tuningThresholds(int a);
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
};

extern const ncclAlgoRing algoRing;
#endif