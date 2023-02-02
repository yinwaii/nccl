#ifndef __MESH_CROSS_H__
#define __MESH_CROSS_H__
#include "base.h"
#include "comm.h"
#include "info.h"
#include "ring.h"

class ncclTopoMeshCross : public ncclTopoBase {
public:
  ncclTopoMeshCross(struct ncclComm *comm);
  ncclResult_t topoPreset(struct ncclTopoRanks *topoRanks);
  ncclResult_t topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks);
  ncclResult_t transportSetup();
};

class ncclEnqueueMeshCross : public ncclEnqueueBase {
private:
  enum Patterns { ncclPatternMeshCross, ncclPatternBroadcast };
  int getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, int nstepsPerLoop) const;

public:
  ncclEnqueueMeshCross() : ncclEnqueueBase("MeshCross") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueRedirect(struct ncclInfo *info) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const;
};

using ncclTuningMeshCross = ncclTuningRing;

#endif