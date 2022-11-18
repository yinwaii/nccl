#ifndef __COLLNET_H__
#define __COLLNET_H__
#include "comm.h"
#include "info.h"
#include "base.h"

class ncclTopoCollNet: public ncclTopoBase {
private:
  ncclResult_t ncclTopoConnectCollNet(int rank);
  int collNetSetup(struct ncclChannel *channel, int rank, int nranks, int masterRank, int masterPeer, int nMasters, int type);
  ncclResult_t checkCollNetSetup(int rank, int collNetSetupFail);

public:
  int *treePatterns;
  ncclTopoCollNet(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
};

class ncclEnqueueCollNet: public ncclEnqueueBase {
private:
  enum Patterns {
    ncclPatternCollTreeUp,
    ncclPatternCollTreeDown
  };

public:
  ncclEnqueueCollNet(): ncclEnqueueBase("Tree") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
  ncclResult_t enqueueChannelThread(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

class ncclTuningCollNet: public ncclTuningBase
{
private:
  // Latencies in us, Bandwidths in GB/s
  const ProtoInfo<float> baseLat = {4.4, 4.4, 0};
  // Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
  const ProtoInfo<float> hwLat[3] = {
      {.5, 1.2, 4.0},   // NVLINK
      {1.0, 1.9, 5.5},  // PCI
      {5.0, 5.0, 10.7}, // NET
  };

public:
  ncclTuningCollNet(ncclComm *comm, std::shared_ptr<ncclTopoBase> topo) : ncclTuningBase(comm, topo) {}
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
};

#endif