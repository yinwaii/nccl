#include "algo_interface.h"

AlgoInfo<ncclTopoAlgo> ncclTopoAlgos(struct ncclComm *comm) {
  AlgoInfo<ncclTopoAlgo> topoAlgos;
  topoAlgos[NCCL_ALGO_TREE] = std::make_shared<ncclTopoTree>(comm);
  topoAlgos[NCCL_ALGO_RING] = std::make_shared<ncclTopoRing>(comm);
  topoAlgos[NCCL_ALGO_COLLNET] = std::make_shared<ncclTopoCollNet>(comm);
  return topoAlgos;
}

ncclResult_t ncclTopoInit(const AlgoInfo<ncclTopoAlgo> &algos) {
  NCCLCHECK(algos[NCCL_ALGO_RING]->graphInit(NCCL_TOPO_PATTERN_RING, 1,
                                             MAXCHANNELS / 2));
  NCCLCHECK(algos[NCCL_ALGO_TREE]->graphInit(
      NCCL_TOPO_PATTERN_SPLIT_TREE,
      1, algos[NCCL_ALGO_RING]->graph.nChannels));
  NCCLCHECK(algos[NCCL_ALGO_COLLNET]->graphInit(
      NCCL_TOPO_PATTERN_TREE, algos[NCCL_ALGO_RING]->graph.nChannels,
      algos[NCCL_ALGO_RING]->graph.nChannels));
  return ncclSuccess;
}

AlgoInfo<ncclTuningAlgo> ncclTuningAlgos(struct ncclComm *comm,
                                         AlgoInfo<ncclTopoAlgo> topoAlgos) {
  AlgoInfo<ncclTuningAlgo> tuningAlgos;
  tuningAlgos[NCCL_ALGO_TREE] =
      std::make_shared<ncclTuningTree>(comm, topoAlgos[NCCL_ALGO_TREE]);
  tuningAlgos[NCCL_ALGO_RING] =
      std::make_shared<ncclTuningRing>(comm, topoAlgos[NCCL_ALGO_RING]);
  tuningAlgos[NCCL_ALGO_COLLNET] =
      std::make_shared<ncclTuningCollNet>(comm, topoAlgos[NCCL_ALGO_COLLNET]);
  return tuningAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclEnqueueAlgos() {
  AlgoInfo<ncclEnqueueAlgo> enqueueAlgos;
  enqueueAlgos[NCCL_ALGO_TREE] = std::make_unique<ncclEnqueueTree>();
  enqueueAlgos[NCCL_ALGO_RING] = std::make_unique<ncclEnqueueRing>();
  enqueueAlgos[NCCL_ALGO_COLLNET] = std::make_unique<ncclEnqueueCollNet>();
  return enqueueAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclAlgos = ncclEnqueueAlgos();
const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
    "Tree", "Ring", "CollNet"};