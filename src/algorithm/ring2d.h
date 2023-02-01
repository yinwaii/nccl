#ifndef __RING_2D_H__
#define __RING_2D_H__
#include "base.h"
#include "comm.h"
#include "info.h"
#include "ring.h"

class ncclTopoRing2D : public ncclTopoBase {
private:
  int *intraRanks, *interRanks;

public:
  ncclTopoRing2D(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks,
                           struct ncclTopoRanks **allTopoRanks);
  ncclResult_t topoDuplicate(int channel);
  ncclResult_t transportSetup();
};

class ncclEnqueueRing2D : public ncclEnqueueBase {
private:
  enum Patterns { ncclPatternRing2D, ncclPatternBroadcast };
  int getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, int nchunksPerLoop, int nstepsPerLoop) const;

public:
  ncclEnqueueRing2D() : ncclEnqueueBase("Ring2D") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueuePattern(struct ncclInfo *info, bool *redirect) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args,
                             struct ncclInfo *info) const;
};

using ncclTuningRing2D = ncclTuningRing;

#endif