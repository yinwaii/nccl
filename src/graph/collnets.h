#ifndef __COLLNET_H__
#define __COLLNET_H__

ncclResult_t ncclTopoPresetCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclTopoRanks* topoRanks);
ncclResult_t ncclTopoPostsetCollNet(struct ncclComm* comm, struct ncclTopoGraph* graph, int* firstRanks, struct ncclTopoRanks** allTopoRanks);
ncclResult_t ncclTransportSetupCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph);
ncclResult_t ncclProxySaveCollCollNetUp(struct ncclProxyArgs *args, int pattern, int root, int nranks);
ncclResult_t ncclProxySaveCollCollNetDn(struct ncclProxyArgs *args, int pattern, int root, int nranks);
ncclResult_t ncclTuningBwCollNet(struct ncclComm *comm, struct ncclTopoGraph *collNetGraph, int coll, int a, int compCap80);
ncclResult_t ncclTuningLatCollNet(struct ncclComm *comm, struct ncclTopoGraph *collNetGraph, int coll, int a);

#endif