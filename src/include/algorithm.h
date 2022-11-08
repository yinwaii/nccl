#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__
#include "comm.h"

int64_t ncclParamCrossNic();

class ncclAlgo {
public:
  struct ncclComm *comm;
  struct ncclTopoGraph graph;
  ncclAlgo() = delete;
  ncclAlgo(struct ncclComm *comm, int crossNic, int collNet, int minChannels, int maxChannels);
  ncclResult_t graphInit(int id, int pattern, ncclTopoSystem *system);
  ncclResult_t graphCopy(struct ncclGraphInfo *dst);
  ncclResult_t graphFit(struct ncclGraphInfo *src);
  virtual ncclResult_t topoPreset(struct ncclTopoRanks* topoRanks) = 0;
  virtual ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) = 0;
  virtual ncclResult_t transportSetup() = 0;
  virtual ncclResult_t ncclProxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks) = 0;
  virtual ncclResult_t tuningBw(int coll, int a, int compCap80) = 0;
  virtual ncclResult_t tuningLat(int coll, int a) = 0;
  virtual ncclResult_t tuningMaxThreads(int a) = 0;
  virtual ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) = 0;
  virtual ncclResult_t tuningThresholds(int a) = 0;
};
class ncclAlgoRing : public ncclAlgo
{
private:
  int *rings;
  ncclResult_t connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks);
  void dumpLine(int *values, int nranks, const char *prefix);
  ncclResult_t ncclBuildRings(int nrings, int *rings, int rank, int nranks, int *prev, int *next);

public:
  ncclAlgoRing() = delete;
  ncclAlgoRing(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t ncclProxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
  ncclResult_t tuningThresholds(int a);
};

class ncclAlgoTree: public ncclAlgo
{
public:
  ncclAlgoTree() = delete;
  ncclAlgoTree(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t ncclProxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
  ncclResult_t tuningThresholds(int a);
};

class ncclAlgoCollNet: public ncclAlgo {
public:
  ncclAlgoCollNet() = delete;
  ncclAlgoCollNet(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t ncclProxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
  ncclResult_t tuningThresholds(int a);
};
#endif