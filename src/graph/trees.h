/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TREES_H_
#define NCCL_TREES_H_

ncclResult_t ncclTopoPresetTree(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclTopoPostsetTree(struct ncclComm* comm, struct ncclTopoGraph* graph, int* firstRanks, struct ncclTopoRanks** allTopoRanks);
ncclResult_t ncclTransportSetupTree(struct ncclComm* comm, struct ncclTopoGraph* graph);
ncclResult_t ncclProxySaveCollTreeUp(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t ncclProxySaveCollTreeDn(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t ncclTuningBwTree(struct ncclComm *comm, struct ncclTopoGraph *graph, int coll, int a, int compCap80);
ncclResult_t ncclTuningLatTree(struct ncclComm* comm, struct ncclTopoGraph* graph, int coll, int a);
ncclResult_t ncclTuningAlgoTimeTree(struct ncclInfo *info, int algorithm, int protocol, float *time);

#endif
