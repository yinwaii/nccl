#ifndef __TREE_H__
#define __TREE_H__
#include "comm.h"
#include "info.h"
#include "algorithm.h"

class ncclAlgoTree: public ncclAlgoBase
{
private:
  enum Patterns {
    ncclPatternTreeUp,
    ncclPatternTreeDown,
    ncclPatternTreeUpDown,
  };
  ncclResult_t connectTrees(int *treeUpRecv, int *treeUpSend, int *treeDnRecv, int *treeDnSend, int *firstRanks);
  ncclResult_t getIndexes(int *ranks, int *indexes, int nNodes, int *firstRanks);
  ncclResult_t setTreeUp(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int u0, int u1);
  ncclResult_t addRanksDown(int *down, int *indexes, int r0, int r1);
  ncclResult_t setTreeDown(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int d0_0, int d0_1, int d1_0, int d1_1);
  ncclResult_t openRing(struct ncclTree *tree, int rank, int upRank);
  ncclResult_t ncclGetBtree(int nranks, int rank, int *u, int *d0, int *d1);
  ncclResult_t ncclGetDtree(int nranks, int rank, int *s0, int *d0_0, int *d0_1, int *s1, int *d1_0, int *d1_1);

public:
  ncclAlgoTree(int maxChannel = MAXCHANNELS/2);
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
  ncclResult_t tuningBw(int coll, int a, int compCap80);
  ncclResult_t tuningLat(int coll, int a);
  ncclResult_t tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
};

extern const ncclAlgoTree algoTree;

#endif