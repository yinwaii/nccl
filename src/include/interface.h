#ifndef __INTERFACE_H__
#define __INTERFACE_H__
#include "comm.h"
#include "algo_interface.h"
ncclResult_t ncclTopoPreset(struct ncclComm *comm, AlgoInfo<ncclTopoAlgo> algos, struct ncclTopoRanks *topoRanks);
ncclResult_t ncclTopoPostset(struct ncclComm *comm, AlgoInfo<ncclTopoAlgo> algos, int *firstRanks, struct ncclTopoRanks **allTopoRanks);
ncclResult_t ncclTopoTuneModel(struct ncclComm *comm, int minCompCap, int maxCompCap, AlgoInfo<ncclTopoAlgo> algos);
ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time);
#endif