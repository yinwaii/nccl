#include "ring.h"
#include "tree.h"
#include "collnet.h"
#include "algo_interface.h"

AlgoInfo<ncclTopoAlgo> ncclTopoAlgos(struct ncclComm *comm) {
	AlgoInfo<ncclTopoAlgo> topoAlgos;
	topoAlgos[NCCL_ALGO_TREE] = std::make_shared<ncclTopoTree>(comm);
	topoAlgos[NCCL_ALGO_RING] = std::make_shared<ncclTopoRing>(comm);
	topoAlgos[NCCL_ALGO_COLLNET] = std::make_shared<ncclTopoCollNet>(comm);
	topoAlgos[NCCL_ALGO_BUTTERFLY] = std::make_shared<ncclTopoButterfly>(comm);
	topoAlgos[NCCL_ALGO_BUTTERFLY2] = std::make_shared<ncclTopoButterfly2>(comm);
	topoAlgos[NCCL_ALGO_BUTTERFLY_YZ] = std::make_shared<ncclTopoButterfly_yz>(comm);
	return topoAlgos;
}

ncclResult_t ncclTopoInit(const AlgoInfo<ncclTopoAlgo> &algos, int tmpNnodes) {
	NCCLCHECK(algos[NCCL_ALGO_RING]->graphInit(NCCL_TOPO_PATTERN_RING, 1, MAXCHANNELS / 2));
	NCCLCHECK(algos[NCCL_ALGO_TREE]->graphInit(tmpNnodes <= 2 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE, 1, algos[NCCL_ALGO_RING]->graph.nChannels));
	NCCLCHECK(algos[NCCL_ALGO_COLLNET]->graphInit(NCCL_TOPO_PATTERN_TREE, algos[NCCL_ALGO_RING]->graph.nChannels, algos[NCCL_ALGO_RING]->graph.nChannels));
	NCCLCHECK(algos[NCCL_ALGO_BUTTERFLY]->graphInit(NCCL_TOPO_PATTERN_RING, 1, algos[NCCL_ALGO_RING]->graph.nChannels));
	NCCLCHECK(algos[NCCL_ALGO_BUTTERFLY2]->graphInit(NCCL_TOPO_PATTERN_RING, 1, algos[NCCL_ALGO_RING]->graph.nChannels));
	NCCLCHECK(algos[NCCL_ALGO_BUTTERFLY_YZ]->graphInit(NCCL_TOPO_PATTERN_BUTTERFLY, 1, MAXCHANNELS));
	return ncclSuccess;
}

AlgoInfo<ncclTuningAlgo> ncclTuningAlgos(struct ncclComm *comm, AlgoInfo<ncclTopoAlgo> topoAlgos) {
	AlgoInfo<ncclTuningAlgo> tuningAlgos;
	tuningAlgos[NCCL_ALGO_TREE] = std::make_shared<ncclTuningTree>(comm, topoAlgos[NCCL_ALGO_TREE]);
	tuningAlgos[NCCL_ALGO_RING] = std::make_shared<ncclTuningRing>(comm, topoAlgos[NCCL_ALGO_RING]);
	tuningAlgos[NCCL_ALGO_COLLNET] = std::make_shared<ncclTuningCollNet>(comm, topoAlgos[NCCL_ALGO_COLLNET]);
	tuningAlgos[NCCL_ALGO_BUTTERFLY] = std::make_shared<ncclTuningButterfly>(comm, topoAlgos[NCCL_ALGO_BUTTERFLY]);
	tuningAlgos[NCCL_ALGO_BUTTERFLY2] = std::make_shared<ncclTuningButterfly2>(comm, topoAlgos[NCCL_ALGO_BUTTERFLY2]);
	tuningAlgos[NCCL_ALGO_BUTTERFLY_YZ] = std::make_shared<ncclTuningButterfly_yz>(comm, topoAlgos[NCCL_ALGO_BUTTERFLY_YZ]);
	return tuningAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclEnqueueAlgos() {
	AlgoInfo<ncclEnqueueAlgo> enqueueAlgos;
	enqueueAlgos[NCCL_ALGO_TREE] = std::make_unique<ncclEnqueueTree>();
	enqueueAlgos[NCCL_ALGO_RING] = std::make_unique<ncclEnqueueRing>();
	enqueueAlgos[NCCL_ALGO_COLLNET] = std::make_unique<ncclEnqueueCollNet>();
	enqueueAlgos[NCCL_ALGO_BUTTERFLY] = std::make_unique<ncclEnqueueButterfly>();
	enqueueAlgos[NCCL_ALGO_BUTTERFLY2] = std::make_unique<ncclEnqueueButterfly2>();
	enqueueAlgos[NCCL_ALGO_BUTTERFLY_YZ] = std::make_unique<ncclEnqueueButterfly_yz>();
	return enqueueAlgos;
}

AlgoInfo<ncclEnqueueAlgo> ncclAlgos = ncclEnqueueAlgos();
const char *ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {"Tree", "Ring", "CollNet", "Butterfly", "Butterfly2", "Butterfly_yz"};