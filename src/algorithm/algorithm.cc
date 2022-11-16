#include "ring.h"
#include "tree.h"
#include "collnet.h"

const ncclAlgoBase *ncclAlgos[NCCL_NUM_ALGORITHMS] = {&algoTree, &algoRing, &algoCollNet};