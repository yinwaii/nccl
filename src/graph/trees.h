/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

ncclResult_t ncclTopoPresetTree(struct ncclComm* comm, struct ncclTopoGraph* treeGraph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclTopoPostsetTree(struct ncclComm* comm, int* firstRanks, struct ncclTopoRanks** allTopoRanks);
ncclResult_t ncclTransportSetupTree(struct ncclComm* comm, struct ncclTopoGraph* treeGraph);

#endif
