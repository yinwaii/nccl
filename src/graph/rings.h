/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __RINGS_H__
#define __RINGS_H__

ncclResult_t ncclTopoPresetRing(struct ncclComm* comm, struct ncclTopoGraph* ringGraph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclTopoPostsetRing(struct ncclComm* comm, int* firstRanks, struct ncclTopoRanks** allTopoRanks);
ncclResult_t ncclTransportSetupRing(struct ncclComm* comm, struct ncclTopoGraph* ringGraph);
ncclResult_t ncclProxySaveCollRing(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t ncclTuningBwRing(struct ncclComm* comm, struct ncclTopoGraph* ringGraph, int coll, int compCap80, int nsteps, float* bandwidths);

#endif