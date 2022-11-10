/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "tuning.h"
#include "topo.h"
#include "info.h"

#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN+5*MAXWIDTH)

int *rings;

void dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(NCCL_INIT,"%s", line);
}

static ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[30];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

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

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
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

ncclResult_t ncclTopoPresetRing(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    int* ringIntra = graph->intra+c*localRanks;
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

ncclResult_t ncclTopoPostsetRing(struct ncclComm* comm, struct ncclTopoGraph* graph, int* firstRanks, struct ncclTopoRanks** allTopoRanks) {
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
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
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

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
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

ncclResult_t ncclTransportSetupRing(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECK(setupChannel(comm, c, comm->rank, comm->nRanks, rings+c*comm->nRanks));
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pSetup(comm, graph, channel, 1, &channel->ring.prev, 1, &channel->ring.next));
  }
  return ncclSuccess;
}

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
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

ncclResult_t ncclProxySaveCollRing(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  struct ncclRing* ring = &args->channel->ring;
  if (NeedProxy(RECV, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxyRecv>(ring->prev, args));
  if (NeedProxy(SEND, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxySend>(ring->next, args));
  return ncclSuccess;
}

ncclResult_t ncclTuningBwRing(struct ncclComm *comm, struct ncclTopoGraph *graph, int coll, int a, int compCap80) {
  int nsteps = coll == ncclCollAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nRanks-1 :
    comm->nRanks;
  float speed = comm->nNodes <= 2 ? graph->speedIntra : graph->speedInter;
  float busBw = graph->nChannels * speed, LL128BusBw = ll128MaxBwPerCh[coll]*graph->nChannels;
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

ncclResult_t ncclTuningLatRing(struct ncclComm* comm, struct ncclTopoGraph* graph, int coll, int a) {
  int nsteps = coll == ncclCollAllReduce ? 2*(comm->nRanks-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nRanks-1 :
    comm->nRanks;
  int nInterSteps = coll == ncclCollAllReduce ? 2*(comm->nNodes-1) :
    coll == ncclCollReduceScatter || coll == ncclCollAllGather ? comm->nNodes-1 :
    comm->nNodes;
  int intraHw = graph->typeIntra == LINK_NVL ? NCCL_HW_NVLINK : NCCL_HW_PCI;
  int hw = comm->nNodes == 1 ? intraHw : NCCL_HW_NET;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->latencies[coll][a][p] = baseLat[a][p];
    float intraLat = hwLat[intraHw][a][p];
    float interLat = hwLat[NCCL_HW_NET][a][p];
    if (comm->nNodes > 1 && p == NCCL_PROTO_LL) intraLat *= 1.8;
    float lat = hwLat[hw][a][p];
    if ((coll == ncclCollReduce || coll == ncclCollBroadcast)) {
      if (graph->sameChannels) {
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

ncclResult_t ncclTuningMaxThreadsRing(struct ncclComm *comm, struct ncclTopoGraph *graph, int a) {
  int simpleDefaultThreads = (graph->speedIntra * graph->nChannels <= PCI_WIDTH) ? 256 : NCCL_MAX_NTHREADS;
  comm->maxThreads[a][NCCL_PROTO_SIMPLE] =
      getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_MAX_NTHREADS, simpleDefaultThreads);
  comm->maxThreads[a][NCCL_PROTO_LL] = getNthreads("NCCL_NTHREADS", ncclParamNthreads(), 2 * WARP_SIZE, NCCL_MAX_NTHREADS, NCCL_MAX_NTHREADS);
  comm->maxThreads[a][NCCL_PROTO_LL128] =
      getNthreads("NCCL_LL128_NTHREADS", ncclParamLl128Nthreads(), NCCL_LL128_MAX_NTHREADS / 4, NCCL_LL128_MAX_NTHREADS, NCCL_LL128_MAX_NTHREADS);
  return ncclSuccess;
}

ncclResult_t ncclTuningAlgoTimeRing(struct ncclInfo* info, int algorithm, int protocol, float* time) {
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

ncclResult_t ncclTuningThresholdsRing(struct ncclComm *comm, int a) {
  ncclTuningThresholds(comm, a);
  comm->threadThresholds[a][NCCL_PROTO_LL] *= comm->nRanks;
  return ncclSuccess;
}