#include "algo_interface.h"
#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN + 5 * MAXWIDTH)

// Topo

ncclTopoRing::ncclTopoRing(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_RING, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoRing::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    int* ringIntra = graph.intra+c*localRanks;
    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
        channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoRing::connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nRanks;
    int* send = ringSend+c*comm->nRanks;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[firstRanks[n]];
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]];
      prev[recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[sendRank] = nextRecvRank;
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        channel1->ring.next = nextRecvRank;
      }
    }
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }
  return ncclSuccess;
}

void ncclTopoRing::dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(NCCL_INIT,"%s", line);
}

ncclResult_t ncclTopoRing::ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[40];
    sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    sprintf(prefix, "Channel %02d/%02d : ", r, nrings);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoRing::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
    }
  }

  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  // Duplicate rings for ncclBuildRing
  memcpy(rings+nChannels*nranks, rings, nChannels*nranks*sizeof(int));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);

  return ncclSuccess;
}

ncclResult_t ncclTopoRing::setupChannel(int channelId, int rank, int nranks, int* ringRanks) {
  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoRing::transportSetup() {
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECK(setupChannel(c, comm->rank, comm->nRanks, rings+c*comm->nRanks));
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next));
  }
  NCCLCHECK(ncclTransportP2pSetup(comm, &graph));
  if (rings == nullptr)
    return ncclInternalError;
  free(rings);
  return ncclSuccess;
}

// Enqueue

ncclResult_t ncclEnqueueRing::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->tuning[algorithm].bandwidths[info->coll][protocol];
  float lat = info->comm->tuning[algorithm].latencies[info->coll][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  if (info->nChannels != 0) bw = bw / info->comm->nChannels * info->nChannels;
  if (protocol == NCCL_PROTO_SIMPLE && info->comm->nNodes > 1 && info->coll == ncclFuncAllReduce 
      && info->nBytes >= info->comm->nRanks/16.0*65536) lat *= 1.9; // Plateau effect of ring
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing::getPattern(int coll, int *pattern) const {
  switch (coll) {
    case ncclFuncBroadcast:
      *pattern = ncclPatternPipelineFrom;
      break;
    case ncclFuncReduce:
      *pattern = ncclPatternPipelineTo;
      break;
    case ncclFuncReduceScatter:
    case ncclFuncAllGather:
      *pattern = ncclPatternRing;
      break;
    case ncclFuncAllReduce:
      *pattern = ncclPatternRingTwice;
      break;
    default:
      *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
      info->nstepsPerLoop = info->nchunksPerLoop = 1; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      WARN("Unknown pattern %d\n", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

bool ncclEnqueueRing::NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) const {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == proxyRecv ?   myrank : nextrank ):
      /* reduce */ (type == proxyRecv ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

ncclResult_t ncclEnqueueRing::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo* info) const {
  int pattern = info->pattern;
  struct ncclRing* ring = &args->channel->ring;
  if (NeedProxy(proxyRecv, pattern, info->root, ring, info->comm->nRanks))
    NCCLCHECK(SaveProxy(proxyRecv, ring->prev, args));
  if (NeedProxy(proxySend, pattern, info->root, ring, info->comm->nRanks))
    NCCLCHECK(SaveProxy(proxySend, ring->next, args));
  return ncclSuccess;
}

ncclResult_t ncclTuningRing::tuningBw(int coll, int a, int compCap80) {
  int nsteps = coll == ncclFuncAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? comm->nRanks-1 :
    comm->nRanks;
  // Convert bus BW to algorithm BW
  float ratio = (1.0 * comm->nRanks) / nsteps;
  float LLRatio = (comm->nNodes > 1 || coll == ncclFuncAllReduce || coll == ncclFuncReduce) ? 1.0 / 4.0 : 1.0 / 3.0;
  // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount
  float LL128Ratio = (float)comm->nRanks / comm->nNodes < 2 ? 0.7 : 0.92 /*120.0/128.0*/;

  float speed = comm->nNodes <= 2 ? topo->graph.speedIntra : topo->graph.speedInter;
  float busBw = topo->graph.nChannels * speed;
  // Various model refinements
  if (compCap80) busBw = std::min(busBw, 235.0f);
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  int index2 = comm->nNodes <= 2 ? comm->nNodes-1 : 2;
  // LL: for single node, we look at GPU type; for multi-node, we look at CPU type
  int index1 = comm->nNodes == 1 ? compCap80 : cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD ? 1 : 0;
  float llMaxBw = llMaxBws[index1][index2], LL128BusBw = ll128MaxBwPerCh[coll] * topo->graph.nChannels;

  comm->tuning[a].bandwidths[coll][NCCL_PROTO_SIMPLE] = busBw * ratio;
  comm->tuning[a].bandwidths[coll][NCCL_PROTO_LL] = std::min(busBw * LLRatio, llMaxBw) * ratio;
  comm->tuning[a].bandwidths[coll][NCCL_PROTO_LL128] = std::min(busBw * LL128Ratio, LL128BusBw) * ratio;
  return ncclSuccess;
}

// Tuning

ncclResult_t ncclTuningRing::tuningLat(int coll, int a) {
  int nsteps = coll == ncclFuncAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? comm->nRanks-1 :
    comm->nRanks;
  int nInterSteps = coll == ncclFuncAllReduce ? 2*(comm->nNodes-1) :
    coll == ncclFuncReduceScatter || coll == ncclFuncAllGather ? comm->nNodes-1 :
    comm->nNodes;
  int intraHw = topo->graph.typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  int hw = comm->nNodes == 1 ? intraHw : NCCL_HW_NET;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->tuning[a].latencies[coll][p] = baseLat[p];
    float intraLat = hwLat[intraHw][p];
    float interLat = hwLat[NCCL_HW_NET][p];
    if (comm->nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
    float lat = hwLat[hw][p];
    if ((coll == ncclFuncReduce || coll == ncclFuncBroadcast)) {
      if (topo->graph.sameChannels) {
        comm->tuning[a].latencies[coll][p] += lat;
      } else {
        if (p == NCCL_PROTO_SIMPLE) lat = hwLatTree[hw][p]; // Add some chunk latency, waiting for proper chunk modeling
        comm->tuning[a].latencies[coll][p] += nsteps * lat;
      }
    } else {
      comm->tuning[a].latencies[coll][p] += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTuningRing::tuningMaxThreads(int a) {
  this->ncclTuningBase::tuningMaxThreads(a);
  int simpleDefaultThreads = (topo->graph.speedIntra * topo->graph.nChannels <= PCI_WIDTH) ? 256 : NCCL_SIMPLE_MAX_NTHREADS;
  comm->tuning[a].maxThreads[NCCL_PROTO_SIMPLE] =
      getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads);
  return ncclSuccess;
}

ncclResult_t ncclTuningRing::tuningThresholds(int a) {
  this->ncclTuningBase::tuningThresholds(a);
  comm->tuning[a].threadThresholds[NCCL_PROTO_LL] *= comm->nRanks;
  return ncclSuccess;
}