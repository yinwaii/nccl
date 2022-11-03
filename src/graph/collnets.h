#ifndef NCCL_COLLNET_H_
#define NCCL_COLLNET_H_

ncclResult_t connectCollNet(struct ncclComm *comm, struct ncclTopoGraph *collNetGraph);
ncclResult_t ncclTopoPresetCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclProxySaveOpCollnetChain(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire);
ncclResult_t ncclProxySaveOpCollnetDirect(struct ncclComm* comm, struct ncclProxyOp* op, bool* justInquire);
ncclResult_t ncclTransportCollnet(struct ncclComm* comm, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoPostsetCollnet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, int* rings);

#endif