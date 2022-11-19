#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__
#include "collnet.h"
#include "ring.h"
#include "tree.h"
#include "algo_config.h"
#include "comm.h"
#include "topo.h"
#include "tuning.h"

using ncclTopoAlgo = std::shared_ptr<ncclTopoBase>;
AlgoInfo<ncclTopoAlgo> ncclTopoAlgos(struct ncclComm *comm);
ncclResult_t ncclTopoInit(const AlgoInfo<ncclTopoAlgo> &algos, int tmpNnodes);
using ncclTuningAlgo = std::shared_ptr<ncclTuningBase>;
AlgoInfo<ncclTuningAlgo> ncclTuningAlgos(struct ncclComm *comm, AlgoInfo<ncclTopoAlgo> topoAlgos);
using ncclEnqueueAlgo = std::unique_ptr<ncclEnqueueBase>;
AlgoInfo<ncclEnqueueAlgo> ncclEnqueueAlgos();

extern AlgoInfo<ncclEnqueueAlgo> ncclAlgos;

#endif