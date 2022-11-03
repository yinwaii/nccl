#ifndef __COLLNET_H__
#define __COLLNET_H__

ncclResult_t ncclTopoPresetCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclTopoPostsetCollNet(struct ncclComm* comm, struct ncclTopoGraph* graph);

#endif