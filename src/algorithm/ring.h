#ifndef __RING_H__
#define __RING_H__
#include "base.h"
#include "comm.h"
#include "info.h"

class ncclTopoRing: public ncclTopoBase {
private:
  ncclResult_t connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks);
  void dumpLine(int *values, int nranks, const char *prefix);
  ncclResult_t ncclBuildRings(int nrings, int *rings, int rank, int nranks, int *prev, int *next);
  ncclResult_t setupChannel(int channelId, int rank, int nranks, int *ringRanks);
public:
  int *rings;
  ncclTopoRing(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ~ncclTopoRing() {}
};

class ncclEnqueueRing: public ncclEnqueueBase {
private:
  enum Patterns {
    ncclPatternRing,
    ncclPatternRingTwice,
    ncclPatternPipelineFrom,
    ncclPatternPipelineTo,
  };
  bool NeedProxy(int type, int pattern, int root, struct ncclRing *ring, int nranks) const;

public:
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem *work) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

class ncclTuningRing: public ncclTuningBase
{
public:
  ncclTuningRing(ncclComm *comm, std::shared_ptr<ncclTopoBase> topo) : ncclTuningBase(comm, topo) {}
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningThresholds(int a);
};

#endif