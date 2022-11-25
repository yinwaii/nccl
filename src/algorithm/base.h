#ifndef __BASE_H__
#define __BASE_H__

#include "comm.h"
#include "info.h"

int64_t ncclParamCrossNic();

class ncclTopoBase {
public:
  struct ncclComm *comm;
  struct ncclTopoGraph graph;
  ncclTopoBase() = delete;
  ncclTopoBase(int id, struct ncclComm *comm, int crossNic, int collNet);
  ncclResult_t graphInit(int pattern, int minChannels, int maxChannels);
  ncclResult_t graphCopy(struct ncclGraphInfo *dst);
  ncclResult_t graphFit(struct ncclGraphInfo *src);
  virtual ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks) = 0;
  virtual ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) = 0;
  virtual ncclResult_t transportSetup() = 0;
  virtual ~ncclTopoBase() {}
};

class ncclEnqueueBase {
public:
  const char *name;
  ncclEnqueueBase(const char *name): name(name) {}
  virtual ncclResult_t getPattern(int coll, int *pattern) const;
  virtual ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  virtual ncclResult_t enqueuePattern(struct ncclInfo *info, bool *redirect) const;
  virtual ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  virtual ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem *work) const;
  virtual ncclResult_t enqueueChannelThread(struct ncclInfo *info) const;
  virtual ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const = 0;
  virtual ~ncclEnqueueBase() {}
};

class ncclTuningBase
{
public:
  ncclComm *comm;
  std::shared_ptr<ncclTopoBase> topo;
  ncclTuningBase() = delete;
  ncclTuningBase(ncclComm *comm, std::shared_ptr<ncclTopoBase> topo): comm(comm), topo(topo){}
  virtual ncclResult_t tuningBw(int coll, int a, int compCap80) = 0;
  virtual ncclResult_t tuningLat(int coll, int a) = 0;
  virtual ncclResult_t tuningMaxThreads(int a);
  virtual ncclResult_t tuningThresholds(int a);
  virtual ~ncclTuningBase() {}
};

#endif