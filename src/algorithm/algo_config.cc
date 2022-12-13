#include "algo_interface.h"

AlgoInfo<ncclTopoAlgo> ncclTopoAlgos(struct ncclComm *comm) {
  AlgoInfo<ncclTopoAlgo> topoAlgos;
  topoAlgos[NCCL_ALGO_RING] = std::make_shared<ncclTopoRing>(comm);
  topoAlgos[NCCL_ALGO_BUTTERFLY2] = std::make_shared<ncclTopoButterfly2>(comm);
  return topoAlgos;
}

ncclResult_t ncclTopoInit(const AlgoInfo<ncclTopoAlgo> &algos) {
  NCCLCHECK(algos[NCCL_ALGO_RING]->graphInit(NCCL_TOPO_PATTERN_RING, 1,
                                             MAXCHANNELS / 2));
  NCCLCHECK(algos[NCCL_ALGO_BUTTERFLY2]->graphInit(
      NCCL_TOPO_PATTERN_RING, 1, algos[NCCL_ALGO_RING]->graph.nChannels));
  return ncclSuccess;
}

AlgoInfo<ncclTuningAlgo> ncclTuningAlgos(struct ncclComm *comm,
                                         AlgoInfo<ncclTopoAlgo> topoAlgos) {
  AlgoInfo<ncclTuningAlgo> tuningAlgos;
  tuningAlgos[NCCL_ALGO_RING] =
      std::make_shared<ncclTuningRing>(comm, topoAlgos[NCCL_ALGO_RING]);
  tuningAlgos[NCCL_ALGO_BUTTERFLY2] = std::make_shared<ncclTuningButterfly2>(
      comm, topoAlgos[NCCL_ALGO_BUTTERFLY2]);
  return tuningAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclEnqueueAlgos() {
  AlgoInfo<ncclEnqueueAlgo> enqueueAlgos;
  enqueueAlgos[NCCL_ALGO_RING] = std::make_unique<ncclEnqueueRing>();
  enqueueAlgos[NCCL_ALGO_BUTTERFLY2] =
      std::make_unique<ncclEnqueueButterfly2>();
  return enqueueAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclAlgos = ncclEnqueueAlgos();
const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {
    "Ring", "Butterfly2"};