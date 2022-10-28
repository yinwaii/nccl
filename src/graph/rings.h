/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_RING_H_
#define NCCL_RING_H_

ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next);
ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks);
ncclResult_t ncclTopoPresetRing(struct ncclComm *comm, struct ncclTopoGraph *ringGraph, struct ncclTopoRanks *topoRanks);

#endif