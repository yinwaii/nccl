#include "../graph/topo.h"
#include "../graph/tuning.h"
#include "algorithm.h"
#include "bootstrap.h"
#include "coll_net.h"
#include "net.h"

const ncclAlgoCollNet algoCollNet;

ncclAlgoCollNet::ncclAlgoCollNet(int maxChannel): ncclAlgoBase(ncclParamCrossNic(), 1) {}

ncclResult_t ncclAlgoCollNet::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->collTree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTree.down[i] = -1;

    int* collNetIntra = graph.intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (collNetIntra[i] == rank) {
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        channel->collTree.up      = collNetIntra[prev];
        channel->collTree.down[0] = collNetIntra[next];
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::ncclTopoConnectCollNet(int rank) {
  int nranks = comm->nRanks;
  int depth = nranks/comm->nNodes;
  int sendIndex = graph.pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
  int sendEndIndex = (sendIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Set root of collTree to id nranks
    if (rank == graph.intra[sendIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == graph.intra[sendEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  int recvIndex = 0;  // recv GPU index is always 0
  int recvEndIndex = (recvIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+comm->nChannels/2+c;
    // Set root of collTree to id nranks
    if (rank == graph.intra[recvIndex+c*comm->localRanks]) { // is master
      channel->collTree.up = nranks;
    }
    if (rank == graph.intra[recvEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTree.down[0] = -1;
    }
    channel->collTree.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", comm->nChannels/2+c, rank, channel->collTree.up, channel->collTree.down[0]);
  }
  return ncclSuccess;
}

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);

ncclResult_t ncclAlgoCollNet::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport() && graph.nChannels) {
    NCCLCHECK(ncclTopoConnectCollNet(comm->rank));
  }

  return ncclSuccess;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// type: 0 for send, 1 for recv
// return: 0 - unsupported, 1 - supported
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
int ncclAlgoCollNet::collNetSetup(struct ncclChannel* channel, int rank, int nranks,  int masterRank, int masterPeer, int nMasters, int type) {
  int rankInCollNet = -1;
  int supported = 0;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;
  int ret = 1;
  if (isMaster) {
    NCCLCHECK(collNetTransport.canConnect(&ret, comm->topo, &graph, myInfo, peerInfo));
  }

  // send master receives connect info from peer recv master
  if (isMaster && type == 0) {
    NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    INFO(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclPeer* root = channel->peers+nranks;
  struct ncclConnector* conn = (type == 1) ? &root->recv : &root->send;
  struct ncclTransportComm* transportComm = (type == 1) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  if (isMaster && ret > 0) {
    NCCLCHECK(transportComm->setup(comm, &graph, myInfo, peerInfo, &myConnect, conn, channel->id));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == 1) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECK(ncclCalloc(&allConnects, nranks));
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster && ret > 0) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct ncclPeer* devRoot = channel->devPeers+nranks;
    struct ncclConnector* devConn = (type == 1) ? &devRoot->recv : &devRoot->send;
    CUDACHECKGOTO(cudaMemcpy(devConn, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == 1) {
    sendrecvExchange.collNetRank = rankInCollNet;
    memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    INFO(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  if (ret > 0) {
    supported = 1;
  }
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return supported;
}

ncclResult_t ncclAlgoCollNet::checkCollNetSetup(int rank, int collNetSetupFail) {
  int nranks = comm->nRanks;
  // AllGather collNet setup results
  int* allGatherFailures;
  NCCLCHECK(ncclCalloc(&allGatherFailures, nranks));
  allGatherFailures[rank] = collNetSetupFail;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGatherFailures, sizeof(int)));
  for (int i=0; i<nranks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  free(allGatherFailures);
  if (collNetSetupFail) {
    if (rank == 0) WARN("Cannot initialize CollNet, using %s instead", ncclNetName());
    // Free collNet resources
    for (int r=0; r<comm->nChannels; r++) {
      struct ncclChannel* channel = comm->channels+r;
      struct ncclPeer* peer = channel->peers+nranks;
      if (peer->send.transportResources && peer->send.transportComm) NCCLCHECK(peer->send.transportComm->free(peer->send.transportResources));
      if (peer->recv.transportResources && peer->recv.transportComm) NCCLCHECK(peer->recv.transportComm->free(peer->recv.transportResources));
      peer->send.transportResources = NULL; // avoid double free
      peer->recv.transportResources = NULL; // avoid double free
    }
    // Set support to 0
    comm->collNetSupport = 0;
  } else {
    comm->collNetSupport = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::transportSetup() {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport() && graph.nChannels) {
    int logicChannels = comm->nChannels/2;
    int collNetSetupFail = 0;
    const int recvIndex = 0;  // recv GPU index is always 0
    const int sendIndex = graph.pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
    for (int c=0; c<logicChannels; c++) {
      struct ncclChannel* channelRecv = comm->channels+logicChannels+c;
      struct ncclChannel* channelSend = comm->channels+c;
      NCCLCHECK(ncclTransportP2pConnect(comm, channelRecv, 1, &channelRecv->collTree.up, 1, channelRecv->collTree.down));
      NCCLCHECK(ncclTransportP2pConnect(comm, channelSend, 1, channelSend->collTree.down, 1, &channelSend->collTree.up));
      const int recvMaster = graph.intra[c*comm->localRanks+recvIndex];
      const int sendMaster = graph.intra[c*comm->localRanks+sendIndex];
      if (collNetSetup(channelRecv, rank, nranks, recvMaster, sendMaster, comm->nNodes, 1) != 1)
        collNetSetupFail = 1;
      else if (collNetSetup(channelSend, rank, nranks, sendMaster, recvMaster, comm->nNodes, 0) != 1)
        collNetSetupFail = 1;
    }
    // Verify CollNet setup across ranks
    NCCLCHECK(checkCollNetSetup(rank, collNetSetupFail));
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  // Adjust pattern for CollNet based on channel index
  int channelId = info->comm->myParams->gridDim.x % info->comm->nChannels;
  info->pattern = (channelId < info->comm->nChannels / info->nSubChannels) ? ncclPatternCollTreeUp : ncclPatternCollTreeDown;
  if (info->pattern == ncclPatternCollTreeUp) {
    // CollTree up
    struct ncclTree *tree = &args->channel->collTree;
    NCCLCHECK(SaveProxy(proxyRecv, tree->down[0], args));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args));
  }
  if (info->pattern == ncclPatternCollTreeDown) {
    // CollTree down
    struct ncclTree *tree = &args->channel->collTree;
    NCCLCHECK(SaveProxy(proxySend, tree->down[0], args));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::tuningBw(int coll, int a, int compCap80) {
  // Convert bus BW to algorithm BW
  float ratio = .5, LLratio = 1.0/6.0;
  float speed = graph.speedIntra;
  float busBw = graph.nChannels * speed;
  // Various model refinements
  if (compCap80) busBw = std::min(busBw, 235.0f);
  busBw *= .9;

  comm->bandwidths[coll][a][NCCL_PROTO_SIMPLE] = busBw * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL] = busBw * LLratio * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL128] = 0;

  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::tuningLat(int coll, int a) {
  int intraHw = graph.typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->latencies[coll][a][p] = baseLat[a][p];
    float intraLat = hwLat[intraHw][a][p];
    float interLat = hwLat[NCCL_HW_NET][a][p];
    if (comm->nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
    comm->latencies[coll][a][p] +=
        2 * (comm->nRanks/comm->nNodes-1) * intraLat + interLat;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::getPattern(int coll, int *pattern) const {
  switch (coll) {
    case ncclFuncAllReduce:
      *pattern = ncclPatternCollTreeUp; break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
    case ncclPatternCollTreeUp:
    case ncclPatternCollTreeDown:
      info->nSubChannels = 2;
      info->nstepsPerLoop = info->nchunksPerLoop = 1; break;
    default:
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem* work) const {
  switch (info->protocol) {
    case NCCL_PROTO_SIMPLE: {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].collTree.depth*16 && sliceInfo->chunkSize > 131072) sliceInfo->chunkSize /= 2;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].collTree.depth*4 && sliceInfo->chunkSize > 65536) sliceInfo->chunkSize /= 2;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].collTree.depth && sliceInfo->chunkSize > 32768) sliceInfo->chunkSize /= 2;
      // Use lastChunkSize as chunkSize
      work->coll.lastChunkSize = sliceInfo->chunkSize / ncclTypeSize(info->datatype);
      break;
    }
    default: {
      this->ncclAlgoBase::enqueueSlice(info, sliceInfo, work);
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoCollNet::enqueueChannelThread(struct ncclInfo *info) const {
  ncclComm *comm = info->comm;
  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels / 2; // CollNet uses one channel for up and one channel for down
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  while (info->nBytes < nc*nt*threadThreshold) {
    if ((nt % 128) == 0) nt/=2;
    else break;
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) nt += WARP_SIZE; // Extra warp for sync
  info->nChannels = nc;
  info->nThreads = nt;
  return ncclSuccess;
}