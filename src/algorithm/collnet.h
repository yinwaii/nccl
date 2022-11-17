#ifndef __COLLNET_H__
#define __COLLNET_H__
#include "comm.h"
#include "info.h"
#include "base.h"

class ncclAlgoCollNet : public ncclAlgoBase
{
private:
  enum Patterns
  {
    ncclPatternCollTreeUp,
    ncclPatternCollTreeDown
  };
  ncclResult_t ncclTopoConnectCollNet(int rank);
  int collNetSetup(struct ncclChannel *channel, int rank, int nranks, int masterRank, int masterPeer, int nMasters, int type);
  ncclResult_t checkCollNetSetup(int rank, int collNetSetupFail);

public:
	ncclAlgoCollNet(int maxChannel = MAXCHANNELS / 2);
	ncclResult_t getPattern(int coll, int *pattern) const;
	ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
	ncclResult_t enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const;
	ncclResult_t enqueueChannelThread(struct ncclInfo *info) const;
	ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
	ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
	ncclResult_t transportSetup();
	ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
	ncclResult_t tuningBw(int coll, int a, int compCap80);
	ncclResult_t tuningLat(int coll, int a);
};

extern const ncclAlgoCollNet algoCollNet;

#endif