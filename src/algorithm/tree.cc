#include "algorithm.h"
#include "../graph/tuning.h"
#include "../graph/topo.h"

const ncclAlgoTree algoTree;

ncclAlgoTree::ncclAlgoTree(int maxChannel): ncclAlgoBase(ncclParamCrossNic(), 0) {}

ncclResult_t ncclAlgoTree::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->tree.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;

    int* treeIntra = graph.intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = graph.pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = graph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
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
ncclResult_t ncclAlgoTree::ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
  int up, down0, down1;
  int bit;
  for (bit=1; bit<nranks; bit<<=1) {
    if (bit & rank) break;
  }

  if (rank == 0) {
    *u = -1;
    *d0 = -1;
    // Child rank is > 0 so it has to be our child 1, not 0.
    *d1 = nranks > 1 ? bit >> 1 : -1;
    return ncclSuccess;
  }

  up = (rank ^ bit) | (bit << 1);
  // if smaller than the parent, we are his first child, otherwise we're his second
  if (up >= nranks) up = (rank ^ bit);
  *parentChildType = (rank < up) ? 0 : 1;
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
 * For the second tree, we use a mirror tree (if nranks is even)
 *
 * 0---------------8                   3----------------11
 *          ______/ \                 / \______
 *         4         \               /         7
 *       /   \        \             /        /   \
 *     2       6       10         1        5      9
 *    / \     / \     /  \       / \      / \    / \
 *   1   3   5   7   9   11     0   2    4   6  8   10
 *
 * or shift it by one rank (if nranks is odd).
 *
 * 0---------------8            1---------------9
 *          ______/ \______              ______/ \______
 *         4               12           5                0
 *       /   \            /           /   \            /
 *     2       6       10           3       7       11
 *    / \     / \     /  \         / \     / \     /  \
 *   1   3   5   7   9   11       2   4   6   8  10   12
 */
ncclResult_t ncclAlgoTree::ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
  // Second tree ... mirror or shift
  if (nranks % 2 == 1) {
    // shift
    int shiftrank = (rank-1+nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : (u+1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}


ncclResult_t ncclAlgoTree::getIndexes(int *ranks, int *indexes, int nNodes, int *firstRanks) {
  for (int n = 0; n < nNodes; n++) indexes[n] = ranks[firstRanks[n]];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)\n", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::connectTrees(int* treeToParent, int* treeToChild0, int* treeToChild1, int* firstRanks, int* treePatterns) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* ranksToParent, *ranksToChild0, *ranksToChild1;
  NCCLCHECK(ncclCalloc(&ranksToParent, nNodes));
  NCCLCHECK(ncclCalloc(&ranksToChild0, nNodes));
  NCCLCHECK(ncclCalloc(&ranksToChild1, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECK(getIndexes(treeToParent+c*comm->nRanks, ranksToParent, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeToChild0+c*comm->nRanks, ranksToChild0, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeToChild1+c*comm->nRanks, ranksToChild1, nNodes, firstRanks));
     if (comm->rank == ranksToParent[node]) {
       NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ranksToChild0 : ranksToChild1, t0u));
       NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ranksToChild0 : ranksToChild1, t1u));
     }
     if (comm->rank == ranksToChild0[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d0));
       NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d0));
     }
     if (comm->rank == ranksToChild1[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, ranksToParent, t0d1));
       NCCLCHECK(setTreeDown(&channel1->tree, ranksToParent, t1d1));
     }
     if (comm->rank == ranksToParent[node] ||
         comm->rank == ranksToChild0[node] ||
         comm->rank == ranksToChild1[node]) {
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }
     channel0->tree.depth = channel1->tree.depth = depth;
  }
  free(ranksToParent);
  free(ranksToChild0);
  free(ranksToChild1);
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  // Gather data from all ranks
  int *treeToParent, *treeToChild0, *treeToChild1;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&treeToParent, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild0, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild1, nranks*MAXCHANNELS));
  for (int i = 0; i < nranks; i++) {
	  for (int c = 0; c < nChannels; c++) {
      treeToParent[c*nranks+i] = allTopoRanks[i]->treeToParent[c];
      treeToChild0[c*nranks+i] = allTopoRanks[i]->treeToChild0[c];
      treeToChild1[c*nranks+i] = allTopoRanks[i]->treeToChild1[c];
	  }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectTrees(treeToParent, treeToChild0, treeToChild1, firstRanks, treePatterns));

  free(treeToParent);
  free(treeToChild0);
  free(treeToChild1);

  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::transportSetup() {
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pConnect(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up));
    NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down));
  }
  NCCLCHECK(ncclTransportP2pSetup(comm, &graph));
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  int pattern = info->pattern;
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxyRecv, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxySend, tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    struct ncclTree* tree = &args->channel->tree;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy(proxySend, tree->down[i], args));
    NCCLCHECK(SaveProxy(proxyRecv, tree->up, args));
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::tuningBw(int coll, int a, int compCap80) {
  // Convert bus BW to algorithm BW
  float ratio = .5;
  float LLRatio = 1.0 / 3.8;
  float LL128Ratio = comm->nNodes == 1 ? 7.0 / 9.0 : 0.915 /*120.0/128.0*/;

  float speed = comm->nNodes <= 2 ? graph.speedIntra : graph.speedInter;
  float busBw = graph.nChannels * speed;
  // Various model refinements
  if (compCap80) busBw = std::min(busBw, 235.0f);
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = comm->nNodes <= 2 ? comm->nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = comm->nNodes == 1 ? compCap80 : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  float perChMaxTreeBw = perChMaxTreeBws[compCap80][index2];
  busBw = std::min(busBw * .92f, graph.nChannels * perChMaxTreeBw);
  float llMaxBw = llMaxBws[index1][index2], LL128BusBw = ll128MaxBwPerCh[coll] * graph.nChannels;

  comm->bandwidths[coll][a][NCCL_PROTO_SIMPLE] = busBw * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL] = std::min(busBw * LLRatio, llMaxBw) * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL128] = std::min(busBw * LL128Ratio, LL128BusBw) * ratio;

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
  if (logSize < 23) bw *= treeCorrectionFactor[protocol][logSize];
  if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::getPattern(int coll, int *pattern) const {
  switch (coll) {
    case ncclFuncBroadcast:
      *pattern = ncclPatternTreeDown; break;
    case ncclFuncReduce:
      *pattern = ncclPatternTreeUp; break;
    case ncclFuncAllReduce:
      *pattern = ncclPatternTreeUpDown; break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::enqueueLoopInfo(struct ncclInfo *info) const {
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

ncclResult_t ncclAlgoTree::enqueueSlice(struct ncclInfo *info, struct ncclSliceInfo *sliceInfo, struct ncclWorkElem* work) const {
  switch (info->protocol) {
    case NCCL_PROTO_SIMPLE: {
      if (info->pattern == ncclPatternTreeUpDown) {
        // Optimize chunkSize / nSteps
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].tree.depth*8 && sliceInfo->chunkSize > 131072) sliceInfo->chunkSize /= 2;
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].tree.depth*4 && sliceInfo->chunkSize > 65536) sliceInfo->chunkSize /= 2;
        while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < info->comm->channels[0].tree.depth && sliceInfo->chunkSize > 32768) sliceInfo->chunkSize /= 2;
      }
      // Use lastChunkSize as chunkSize
      work->coll.lastChunkSize = sliceInfo->chunkSize / ncclTypeSize(info->datatype);
      break;
    }
    case NCCL_PROTO_LL128: {
      int nNodes = info->comm->nNodes;
      float ppn = info->comm->nRanks / (float)nNodes;
      float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < nstepsLL128*64/ppn && sliceInfo->chunkSize > 131072) sliceInfo->chunkSize /= 2;
      while (info->nBytes / (info->nChannels*sliceInfo->chunkSize) < nstepsLL128*16/ppn && sliceInfo->chunkSize > 32768) sliceInfo->chunkSize /= 2;
      // Use lastChunkSize as chunkSize
      work->coll.lastChunkSize = sliceInfo->chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
      break;
    }
    default: {
      this->ncclAlgoBase::enqueueSlice(info, sliceInfo, work);
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoTree::enqueueChannelThread(struct ncclInfo *info) const {
  this->ncclAlgoBase::enqueueChannelThread(info);
  if (info->protocol == NCCL_PROTO_SIMPLE) 
    info->nThreads += WARP_SIZE;
  return ncclSuccess;
}