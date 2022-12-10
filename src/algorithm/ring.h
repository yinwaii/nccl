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
  ncclEnqueueRing(): ncclEnqueueBase("Ring") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

class ncclTuningRing: public ncclTuningBase
{
private:
  // Latencies in us, Bandwidths in GB/s
  const ProtoInfo<float> baseLat = { 3.6, 10.0, 8.4 };
  // Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
  const ProtoInfo<float> hwLat[3] = {
    {.47, 1.9, 3.4}, // NVLINK
    {1.0, 2.5, 5.7}, // PCI
    {2.7, 4.0, 9.6}, // NET
  };
  const ProtoInfo<float> hwLatTree[3] = {
      {.52, 1.25, 28}, // NVLINK
      {1.0, 1.9, 28},  // PCI
      {5.0, 8.5, 28},  // NET
  };

public:
  ncclTuningRing(ncclComm *comm, std::shared_ptr<ncclTopoBase> topo) : ncclTuningBase(comm, topo) {}
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningThresholds(int a);
};

#endif