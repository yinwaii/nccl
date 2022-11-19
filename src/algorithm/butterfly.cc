#include "algo_interface.h"

// Topo

ncclTopoTree::ncclTopoTree(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_TREE, comm, ncclParamCrossNic(), 0) {}

