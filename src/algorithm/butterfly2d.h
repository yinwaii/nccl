#ifndef __BUTTERFLY_2D_H__
#define __BUTTERFLY_2D_H__
#include "base.h"
#include "comm.h"
#include "info.h"
#include "ring.h"

class ncclTopoButterfly2D : public ncclTopoBase {
private:
  int *peerRanks, *intraRanks;

public:
  ncclTopoButterfly2D(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks,
                           struct ncclTopoRanks **allTopoRanks);
  ncclResult_t topoDuplicate(int channel);
  ncclResult_t transportSetup();
};

class ncclEnqueueButterfly2D : public ncclEnqueueBase {
private:
  enum Patterns { ncclPatternButterfly2D, ncclPatternBroadcast };
	int getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, size_t size, int nstepsPerLoop = 1) const;

public:
  ncclEnqueueButterfly2D() : ncclEnqueueBase("Butterfly2D") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueRedirect(struct ncclInfo *info) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args,
                             struct ncclInfo *info) const;
};

using ncclTuningButterfly2D = ncclTuningRing;

#endif