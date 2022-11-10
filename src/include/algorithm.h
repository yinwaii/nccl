#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__
#include "comm.h"
#include "info.h"

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
  virtual ncclResult_t proxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks) = 0;
  virtual ncclResult_t tuningBw(int coll, int a, int compCap80) = 0;
  virtual ncclResult_t tuningLat(int coll, int a) = 0;
  virtual ncclResult_t tuningMaxThreads(int a);
  virtual ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
  virtual ncclResult_t tuningThresholds(int a);
  virtual ~ncclAlgo();
};
class ncclAlgoRing : public ncclAlgo
{
private:
  ncclResult_t connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks);
  void dumpLine(int *values, int nranks, const char *prefix);
  ncclResult_t ncclBuildRings(int nrings, int *rings, int rank, int nranks, int *prev, int *next);
  ncclResult_t setupChannel(int channelId, int rank, int nranks, int *ringRanks);
  bool NeedProxy(int type, int pattern, int root, struct ncclRing *ring, int nranks);

public:
  int *rings;
  ncclAlgoRing() = delete;
  ncclAlgoRing(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningMaxThreads(int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
  ncclResult_t tuningThresholds(int a);
};

class ncclAlgoTree: public ncclAlgo
{
private:
  ncclResult_t connectTrees(int *treeUpRecv, int *treeUpSend, int *treeDnRecv, int *treeDnSend, int *firstRanks);
  ncclResult_t getIndexes(int *ranks, int *indexes, int nNodes, int *firstRanks);
  ncclResult_t setTreeUp(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int u0, int u1);
  ncclResult_t addRanksDown(int *down, int *indexes, int r0, int r1);
  ncclResult_t setTreeDown(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int d0_0, int d0_1, int d1_0, int d1_1);
  ncclResult_t openRing(struct ncclTree *tree, int rank, int upRank);
  ncclResult_t ncclGetBtree(int nranks, int rank, int *u, int *d0, int *d1);
  ncclResult_t ncclGetDtree(int nranks, int rank, int *s0, int *d0_0, int *d0_1, int *s1, int *d1_0, int *d1_1);

public:
  ncclAlgoTree() = delete;
  ncclAlgoTree(struct ncclComm *comm, int maxChannel = MAXCHANNELS/2);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
};

class ncclAlgoCollNet: public ncclAlgo {
private:
  ncclResult_t ncclTopoConnectCollNet(int rank);
  int collNetSetup(struct ncclChannel *channel, int rank, int nranks, int masterRank, int masterPeer, int nMasters, int type);
  ncclResult_t checkCollNetSetup(int rank, int collNetSetupFail);

public:
  ncclAlgoCollNet() = delete;
  ncclAlgoCollNet(struct ncclComm *comm, int maxChannel = MAXCHANNELS/2);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks);
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
};
#endif