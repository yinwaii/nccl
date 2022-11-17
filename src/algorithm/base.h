#ifndef __BASE_H__
#define __BASE_H__

#include "comm.h"
#include "info.h"

int64_t ncclParamCrossNic();

class ncclAlgoBase {
public:
  struct ncclComm *comm;
  struct ncclTopoGraph graph;
  ncclAlgoBase() = delete;
  ncclAlgoBase(int crossNic, int collNet);
  virtual ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t graphInit(struct ncclComm *comm, int id, int pattern, ncclTopoSystem *system, int minChannels, int maxChannels);
  ncclResult_t graphCopy(struct ncclGraphInfo *dst);
  ncclResult_t graphFit(struct ncclGraphInfo *src);
  virtual ncclResult_t enqueuePattern(struct ncclInfo* info) const;
  virtual ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  virtual ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
  virtual ncclResult_t enqueueChannelThread(struct ncclInfo *info) const;
  virtual ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks) = 0;
  virtual ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) = 0;
  virtual ncclResult_t transportSetup() = 0;
  virtual ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const = 0;
  virtual ncclResult_t tuningBw(int coll, int a, int compCap80) = 0;
  virtual ncclResult_t tuningLat(int coll, int a) = 0;
  virtual ncclResult_t tuningMaxThreads(int a);
  virtual ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  virtual ncclResult_t tuningThresholds(int a);
  virtual ~ncclAlgoBase();
};

#endif