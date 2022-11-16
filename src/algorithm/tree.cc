#include "algorithm.h"
#include "../graph/tuning.h"
#include "../graph/topo.h"

const ncclAlgoTree algoTree;

ncclAlgoTree::ncclAlgoTree(int maxChannel): ncclAlgo(ncclParamCrossNic(), 0) {}

ncclResult_t ncclAlgoTree::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->treeUp.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeUp.down[i] = -1;
    channel->treeDn.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeDn.down[i] = -1;

    int* treeIntra = graph.intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (treeIntra[i] == rank) {
        int recvIndex = 0, sendIndex = graph.pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        // Tree loop always flows in the same direction. Other trees are symmetric, i.e.
        // up/down go in reverse directions
        int sym = graph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP ? 0 : 1;

        // Down tree is common
        topoRanks->treeDnRecv[c] = treeIntra[recvIndex];
        topoRanks->treeDnSend[c] = treeIntra[sendIndex];
        channel->treeDn.up       = treeIntra[prev];
        channel->treeDn.down[0]  = treeIntra[next];
        // Up tree depends on the pattern
        topoRanks->treeUpRecv[c] = sym ? topoRanks->treeDnSend[c] : topoRanks->treeDnRecv[c];
        topoRanks->treeUpSend[c] = sym ? topoRanks->treeDnRecv[c] : topoRanks->treeDnSend[c];
        channel->treeUp.down[0]  = sym ? channel->treeDn.down[0]  : channel->treeDn.up ;
        channel->treeUp.up       = sym ? channel->treeDn.up       : channel->treeDn.down[0];
      }
    }
  }
  
  return ncclSuccess;
}

/* Btree which alternates leaves and nodes.
 * Assumes root is 0, which conveniently builds a tree on powers of two,
 * (because we have pow2-1 ranks) which lets us manipulate bits.
 * Find first non-zero bit, then :
 * Find the parent :
 *   xx01[0] -> xx10[0] (1,5,9 below) or xx00[0] if xx10[0] is out of bounds (13 below)
 *   xx11[0] -> xx10[0] (3,7,11 below)
 * Find the children :
 *   xx10[0] -> xx01[0] (2,4,6,8,10,12) or -1 (1,3,5,7,9,11,13)
 *   xx10[0] -> xx11[0] (2,4,6,8,10) or xx101[0] (12) or xx1001[0] ... or -1 (1,3,5,7,9,11,13)
 *
 * Illustration :
 * 0---------------8
 *          ______/ \______
 *         4               12
 *       /   \            /  \
 *     2       6       10     \
 *    / \     / \     /  \     \
 *   1   3   5   7   9   11    13
 */
ncclResult_t ncclAlgoTree::ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1) {
  int up, down0, down1;
  int bit;
  for (bit=1; bit<nranks; bit<<=1) {
    if (bit & rank) break;
  }

  if (rank == 0) {
    *u = -1;
    *d0 = nranks > 1 ? bit >> 1 : -1;
    *d1 = -1;
    return ncclSuccess;
  }

  up = (rank ^ bit) | (bit << 1);
  if (up >= nranks) up = (rank ^ bit);
  *u = up;

  int lowbit = bit >> 1;
  // down0 is always within bounds
  down0 = lowbit == 0 ? -1 : rank-lowbit;

  down1 = lowbit == 0 ? -1 : rank+lowbit;
  // Make sure down1 is within bounds
  while (down1 >= nranks) {
    down1 = lowbit == 0 ? -1 : rank+lowbit;
    lowbit >>= 1;
  }
  *d0 = down0; *d1 = down1;

  return ncclSuccess;
}

/* Build a double binary tree. Take the previous tree for the first tree.
 * For the second tree, we use a mirror tree (if nranks is odd)
 *
 *                 8---------0---------5
 *          ______/ \______      _____/ \______
 *         4               12   1              9
 *       /   \            /      \           /   \
 *     2       6       10          3       7      10
 *    / \     / \     /  \        / \     / \    /  \
 *   1   3   5   7   9   11      2   4   6   8  11  12
 *
 * or shift it by one rank (if nranks is even)
 *
 *                 8---------0--------------9
 *          ______/ \                ______/ \
 *         4         \              5         \
 *       /   \        \           /   \        \
 *     2       6       10       3       7       11
 *    / \     / \     /  \     / \     / \     /  \
 *   1   3   5   7   9   11   2   4   6   8   10   1
 */
ncclResult_t ncclAlgoTree::ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* s1, int* d1_0, int* d1_1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1);
  // Second tree ... mirror or shift
  if (nranks % 2 == 0) {
    // shift
    int shiftrank = (rank-1+nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1);
    *s1 = u == -1 ? -1 : (u+1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1);
    *s1 = u == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::getIndexes(int *ranks, int *indexes, int nNodes, int *firstRanks) {
  for (int n = 0; n < nNodes; n++)
	indexes[n] = ranks[firstRanks[n]];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::setTreeUp(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int u0, int u1) {
  if (u0 != -1)
	tree0->up = indexes[u0];
  if (u1 != -1)
	tree1->up = indexes[u1];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::addRanksDown(int *down, int *indexes, int r0, int r1) {
  int x = 0;
  if (down[x] >= 0)
	x++;
  if (down[x] >= 0)
  {
	WARN("Internal error : tree already has more than one child (%d %d %d)\n", down[0], down[1], down[2]);
	return ncclInternalError;
  }
  if (r0 != -1)
	down[x++] = indexes[r0];
  if (r1 != -1)
	down[x++] = indexes[r1];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::setTreeDown(struct ncclTree *tree0, struct ncclTree *tree1, int *indexes, int d0_0, int d0_1, int d1_0, int d1_1) {
  NCCLCHECK(addRanksDown(tree0->down, indexes, d0_0, d0_1));
  NCCLCHECK(addRanksDown(tree1->down, indexes, d1_0, d1_1));
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::openRing(struct ncclTree *tree, int rank, int upRank)
{
  if (tree->down[0] == upRank)
	tree->down[0] = -1;
  if (rank == upRank)
	tree->up = -1;
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::connectTrees(int* treeUpRecv, int* treeUpSend, int* treeDnRecv, int* treeDnSend, int* firstRanks) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* indexesSend, *indexesRecv;
  NCCLCHECK(ncclCalloc(&indexesSend, nNodes));
  NCCLCHECK(ncclCalloc(&indexesRecv, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int u0, d0_0, d0_1, u1, d1_0, d1_1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &u0, &d0_0, &d0_1, &u1, &d1_0, &d1_1));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECK(getIndexes(treeUpSend+c*comm->nRanks, indexesSend, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeUpRecv+c*comm->nRanks, indexesRecv, nNodes, firstRanks));
     NCCLCHECK(openRing(&channel0->treeUp, comm->rank, indexesSend[node]));
     NCCLCHECK(openRing(&channel1->treeUp, comm->rank, indexesSend[node]));
     int root = indexesSend[node];
     if (indexesSend[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeUp, &channel1->treeUp, indexesRecv, u0, u1));
     if (indexesRecv[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeUp, &channel1->treeUp, indexesSend, d0_0, d0_1, d1_0, d1_1));
     NCCLCHECK(getIndexes(treeDnSend+c*comm->nRanks, indexesSend, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeDnRecv+c*comm->nRanks, indexesRecv, nNodes, firstRanks));
     NCCLCHECK(openRing(&channel0->treeDn, comm->rank, u0 == -1 ? root : indexesRecv[node]));
     NCCLCHECK(openRing(&channel1->treeDn, comm->rank, u1 == -1 ? root : indexesRecv[node]));
     if (indexesSend[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeDn, &channel1->treeDn, indexesRecv, d0_0, d0_1, d1_0, d1_1));
     if (indexesRecv[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeDn, &channel1->treeDn, indexesSend, u0, u1));
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c,           channel0->treeUp.up, channel0->treeUp.down[0], channel0->treeUp.down[1], channel0->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c+nChannels, channel1->treeUp.up, channel1->treeUp.down[0], channel1->treeUp.down[1], channel1->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c,           channel0->treeDn.up, channel0->treeDn.down[0], channel0->treeDn.down[1], channel0->treeDn.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c+nChannels, channel1->treeDn.up, channel1->treeDn.down[0], channel1->treeDn.down[1], channel1->treeDn.down[2]);
     channel0->treeUp.depth = channel1->treeUp.depth = depth;
  }
  free(indexesSend);
  free(indexesRecv);
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  // Gather data from all ranks
  int *treeUpRecv, *treeUpSend, *treeDnRecv, *treeDnSend;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&treeUpRecv, nranks * MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpSend, nranks * MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnRecv, nranks * MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnSend, nranks * MAXCHANNELS));
  for (int i = 0; i < nranks; i++)
  {
	for (int c = 0; c < nChannels; c++)
	{
	  treeUpRecv[c * nranks + i] = allTopoRanks[i]->treeUpRecv[c];
	  treeUpSend[c * nranks + i] = allTopoRanks[i]->treeUpSend[c];
	  treeDnRecv[c * nranks + i] = allTopoRanks[i]->treeDnRecv[c];
	  treeDnSend[c * nranks + i] = allTopoRanks[i]->treeDnSend[c];
	}
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectTrees(treeUpRecv, treeUpSend, treeDnRecv, treeDnSend, firstRanks));

  free(treeUpRecv);
  free(treeUpSend);
  free(treeDnRecv);
  free(treeDnSend);

  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::transportSetup() {
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, NCCL_MAX_TREE_ARITY, channel->treeUp.down, 1, &channel->treeUp.up));
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->treeDn.up, NCCL_MAX_TREE_ARITY, channel->treeDn.down));
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  int pattern = info->pattern;
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    struct ncclTree* tree = &args->channel->treeUp;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxyRecv>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxySend>(tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    struct ncclTree* tree = &args->channel->treeDn;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxySend>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxyRecv>(tree->up, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::tuningBw(int coll, int a, int compCap80) {
  float speed = comm->nNodes <= 2 ? graph.speedIntra : graph.speedInter;
  float busBw = graph.nChannels * speed, LL128BusBw = ll128MaxBwPerCh[coll] * graph.nChannels * 7.0 / 9.0;
  // Various model refinements
  if (compCap80)
	busBw = std::min(busBw, 235.0f);
  float maxTreeBw = comm->nNodes > 2 ? 80.0 : 110.0;
  float maxTreeBwCompCap80 = comm->nNodes > 2 ? 105.0 : 130.0;
  // Convert bus BW to algorithm BW
  float ratio = .5;
  float LLRatio = 1.0 / 3.8;
  float LL128Ratio = comm->nNodes == 1 ? 7.0 / 9.0 : 0.915 /*120.0/128.0*/;

  comm->bandwidths[coll][a][NCCL_PROTO_SIMPLE] = std::min(busBw * .9f, maxTreeBw) * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL] = std::min(busBw * .9f, maxTreeBw) * LLRatio * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL128] = std::min(std::min(busBw * .9f, compCap80 ? maxTreeBwCompCap80 : maxTreeBw) * LL128Ratio, LL128BusBw) * ratio;

  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::tuningLat(int coll, int a) {
  int intraHw = graph.typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->latencies[coll][a][p] = baseLat[a][p];
    float intraLat = hwLat[intraHw][a][p];
    float interLat = hwLat[NCCL_HW_NET][a][p];
    if (comm->nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
    comm->latencies[coll][a][p] +=
        2 * ((comm->nRanks/comm->nNodes-1) * intraLat + log2i(comm->nNodes) * interLat);
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  int logSize = log2i(info->nBytes>>6);
  if (logSize < 22) bw *= treeCorrectionFactor[protocol][logSize];
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::getPattern(int coll, int *pattern) const {
  switch (coll) {
    case ncclCollBroadcast:
      *pattern = ncclPatternTreeDown; break;
    case ncclCollReduce:
      *pattern = ncclPatternTreeUp; break;
    case ncclCollAllReduce:
      *pattern = ncclPatternTreeUpDown; break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::enqueueLoopInfo(struct ncclInfo *info) const {
  info->nSubChannels = 1;
  switch (info->pattern) {
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
      info->nstepsPerLoop = info->nchunksPerLoop = 1; break;
    default:
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclColl *coll) const {
  switch (info->protocol) {
    case NCCL_PROTO_SIMPLE: {
      if (info->pattern == ncclPatternTreeUpDown) {
        // Optimize chunkSize / nSteps
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].treeUp.depth*8 && sliceInfo->chunkSize > 131072) sliceInfo->chunkSize /= 2;
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].treeUp.depth*4 && sliceInfo->chunkSize > 65536) sliceInfo->chunkSize /= 2;
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].treeUp.depth && sliceInfo->chunkSize > 32768) sliceInfo->chunkSize /= 2;
      }
      // Use lastChunkSize as chunkSize
      coll->args.coll.lastChunkSize = sliceInfo->chunkSize / ncclTypeSize(info->datatype);
      break;
    }
    case NCCL_PROTO_LL128: {
      int nNodes = info->comm->nNodes;
      float ppn = info->comm->nRanks / (float)nNodes;
      float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < nstepsLL128*64/ppn && sliceInfo->chunkSize > 131072) sliceInfo->chunkSize /= 2;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < nstepsLL128*16/ppn && sliceInfo->chunkSize > 32768) sliceInfo->chunkSize /= 2;
      // Use lastChunkSize as chunkSize
      coll->args.coll.lastChunkSize = sliceInfo->chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
      break;
    }
    default: {
      this->ncclAlgo::enqueueSlice(info, sliceInfo, coll);
      break;
    }
  }
  return ncclSuccess;
}