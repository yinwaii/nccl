#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__
#include "comm.h"
#include "ring.h"
#include "tree.h"
#include "collnet.h"

using ncclTopoAlgo = std::shared_ptr<ncclTopoBase>;
AlgoInfo<ncclTopoAlgo> ncclTopoAlgos(struct ncclComm *comm);
using ncclTuningAlgo = std::shared_ptr<ncclTuningBase>;
AlgoInfo<ncclTuningAlgo> ncclTuningAlgos(struct ncclComm *comm, AlgoInfo<ncclTopoAlgo> topoAlgos);
using ncclEnqueueAlgo = std::unique_ptr<ncclEnqueueBase>;
AlgoInfo<ncclEnqueueAlgo> ncclEnqueueAlgos();

extern AlgoInfo<ncclEnqueueAlgo> ncclAlgos;

#endif