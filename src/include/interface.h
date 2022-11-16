#ifndef __INTERFACE_H__
#define __INTERFACE_H__
#include "comm.h"
#include "algorithm.h"
ncclResult_t ncclTopoPreset(struct ncclComm *comm, ncclAlgoBase **algos, struct ncclTopoRanks *topoRanks);
ncclResult_t ncclTopoPostset(struct ncclComm *comm, ncclAlgoBase **algos, int *firstRanks, struct ncclTopoRanks **allTopoRanks);
ncclResult_t ncclTopoTuneModel(struct ncclComm *comm, int minCompCap, int maxCompCap, ncclAlgoBase **algos);
ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
#endif