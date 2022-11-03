#include "comm.h"
#include "coll_net.h"

static ncclResult_t ncclTopoConnectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, int rank) {
  int nranks = comm->nRanks;
  int depth = nranks/comm->nNodes;
  int sendIndex = collNetGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
  int sendEndIndex = (sendIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[sendIndex+c*comm->localRanks]) { // is master
      channel->collTreeUp.up = channel->collTreeDn.up = nranks;
    }
    if (rank == collNetGraph->intra[sendEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTreeUp.down[0] = channel->collTreeDn.down[0] = -1;
    }
    channel->collTreeUp.depth = channel->collTreeDn.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", c, rank, channel->collTreeUp.up, channel->collTreeUp.down[0]);
  }
  int recvIndex = 0;  // recv GPU index is always 0
  int recvEndIndex = (recvIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+comm->nChannels+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[recvIndex+c*comm->localRanks]) { // is master
      channel->collTreeUp.up = channel->collTreeDn.up = nranks;
    }
    if (rank == collNetGraph->intra[recvEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTreeUp.down[0] = channel->collTreeDn.down[0] = -1;
    }
    channel->collTreeUp.depth = channel->collTreeDn.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", comm->nChannels+c, rank, channel->collTreeDn.up, channel->collTreeDn.down[0]);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPresetCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->collTreeUp.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTreeUp.down[i] = -1;
    channel->collTreeDn.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTreeDn.down[i] = -1;

    int* collNetIntra = collNetGraph->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (collNetIntra[i] == rank) {
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        // CollTrees are always symmetric, i.e.
        // up/down go in reverse directions
        channel->collTreeDn.up      = collNetIntra[prev];
        channel->collTreeDn.down[0] = collNetIntra[next];
        channel->collTreeUp.down[0] = channel->collTreeDn.down[0];
        channel->collTreeUp.up      = channel->collTreeDn.up;
      }
    }
  }

  return ncclSuccess;
}

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);

ncclResult_t ncclTopoPostsetCollNet(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport() && graph->nChannels) {
    NCCLCHECK(ncclTopoConnectCollNet(comm, graph, comm->rank));
  }

  return ncclSuccess;
}
