#ifndef __INTERFACE_H__
#define __INTERFACE_H__
#include "comm.h"
#include "algorithm.h"
ncclResult_t ncclTopoPreset(struct ncclComm *comm, ncclAlgo **algos, struct ncclTopoRanks *topoRanks);
ncclResult_t ncclTopoPostset(struct ncclComm *comm, ncclAlgo **algos, int *firstRanks, struct ncclTopoRanks **allTopoRanks);
ncclResult_t ncclTopoTuneModel(struct ncclComm *comm, int minCompCap, int maxCompCap, ncclAlgo **algos);
ncclResult_t ncclTransportSetup(struct ncclComm *comm, ncclAlgo **algos);
#endif