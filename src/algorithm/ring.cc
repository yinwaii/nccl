#include "algorithm.h"
#include "../graph/tuning.h"
#include "../graph/topo.h"
#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN + 5 * MAXWIDTH)

const ncclAlgoRing algoRing;

ncclAlgoRing::ncclAlgoRing(): ncclAlgo(ncclParamCrossNic(), 0) {}

ncclResult_t ncclAlgoRing::topoPreset(struct ncclTopoRanks *topoRanks) {
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

ncclResult_t ncclAlgoRing::connectRings(int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
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

void ncclAlgoRing::dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(NCCL_INIT,"%s", line);
}

ncclResult_t ncclAlgoRing::ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[30];
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

ncclResult_t ncclAlgoRing::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
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

ncclResult_t ncclAlgoRing::setupChannel(int channelId, int rank, int nranks, int* ringRanks) {
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

ncclResult_t ncclAlgoRing::transportSetup() {
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECK(setupChannel(c, comm->rank, comm->nRanks, rings+c*comm->nRanks));
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->ring.prev, 1, &channel->ring.next));
  }
  if (rings == nullptr)
    return ncclInternalError;
  free(rings);
  return ncclSuccess;
}

bool ncclAlgoRing::NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) const {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

ncclResult_t ncclAlgoRing::proxySaveColl(struct ncclProxyArgs *args, int pattern, int root, int nranks) const {
  struct ncclRing* ring = &args->channel->ring;
  if (NeedProxy(RECV, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxyRecv>(ring->prev, args));
  if (NeedProxy(SEND, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxySend>(ring->next, args));
  return ncclSuccess;
}

ncclResult_t ncclAlgoRing::tuningBw(int coll, int a, int compCap80) {
  int nsteps = coll == ncclCollAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nRanks-1 :
    comm->nRanks;
  float speed = comm->nNodes <= 2 ? graph.speedIntra : graph.speedInter;
  float busBw = graph.nChannels * speed, LL128BusBw = ll128MaxBwPerCh[coll]*graph.nChannels;
  // Various model refinements
  if (compCap80) busBw = std::min(busBw, 235.0f);
  // Convert bus BW to algorithm BW
  float ratio = (1.0 * comm->nRanks) / nsteps;
  float LLRatio = (comm->nNodes > 1 || coll == ncclCollAllReduce || coll == ncclCollReduce) ? 1.0/4.0 : 1.0/3.0;
  // if ppn < 2, then we are sending/receiving at the same GPU through the NIC, apply some bw discount
  float LL128Ratio = (float)comm->nRanks / comm->nNodes < 2 ? 0.7 : 0.92 /*120.0/128.0*/;

  comm->bandwidths[coll][a][NCCL_PROTO_SIMPLE] = busBw * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL] = busBw * LLRatio * ratio;
  comm->bandwidths[coll][a][NCCL_PROTO_LL128] = std::min(busBw * LL128Ratio, LL128BusBw) * ratio;

  return ncclSuccess;
}

ncclResult_t ncclAlgoRing::tuningLat(int coll, int a) {
  int nsteps = coll == ncclCollAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nRanks-1 :
    comm->nRanks;
  int nInterSteps = coll == ncclCollAllReduce ? 2*(comm->nNodes-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nNodes-1 :
    comm->nNodes;
  int intraHw = graph.typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  int hw = comm->nNodes == 1 ? intraHw : NCCL_HW_NET;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->latencies[coll][a][p] = baseLat[a][p];
    float intraLat = hwLat[intraHw][a][p];
    float interLat = hwLat[NCCL_HW_NET][a][p];
    if (comm->nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
    float lat = hwLat[hw][a][p];
    if ((coll == ncclCollReduce || coll == ncclCollBroadcast)) {
      if (graph.sameChannels) {
        comm->latencies[coll][a][p] += lat;
      } else {
        if (p == NCCL_PROTO_SIMPLE) lat = hwLat[hw][NCCL_ALGO_TREE][p]; // Add some chunk latency, waiting for proper chunk modeling
        comm->latencies[coll][a][p] += nsteps*lat;
      }
    } else {
      comm->latencies[coll][a][p] += (nsteps-nInterSteps)*intraLat + nInterSteps*interLat;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclAlgoRing::tuningMaxThreads(int a) {
  this->ncclAlgo::tuningMaxThreads(a);
  int simpleDefaultThreads = (graph.speedIntra * graph.nChannels <= PCI_WIDTH) ? 256 : NCCL_MAX_NTHREADS;
  comm->maxThreads[a][NCCL_PROTO_SIMPLE] =
      getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_MAX_NTHREADS, simpleDefaultThreads);
  return ncclSuccess;
}

ncclResult_t ncclAlgoRing::tuningAlgoTime(struct ncclInfo *info, int algorithm, int protocol, float *time) const {
  float bw = info->comm->bandwidths[info->coll][algorithm][protocol];
  float lat = info->comm->latencies[info->coll][algorithm][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  if (protocol == NCCL_PROTO_SIMPLE && info->comm->nNodes > 1 && info->coll == ncclCollAllReduce 
      && info->nBytes >= info->comm->nRanks/16.0*65536) lat *= 1.9; // Plateau effect of ring
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclAlgoRing::tuningThresholds(int a) {
  this->ncclAlgo::tuningThresholds(a);
  comm->threadThresholds[a][NCCL_PROTO_LL] *= comm->nRanks;
  return ncclSuccess;
}