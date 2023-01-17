#include "algo_interface.h"
#include <assert.h>

// Topo

ncclTopoRing2D::ncclTopoRing2D(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_RING_2D, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoRing2D::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  intraRanks = new int[localRanks * MAXCHANNELS];

  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    int *ring2DIntra = graph.intra + c * localRanks;
    for (int i = 0; i < localRanks; i++) {
      if (ring2DIntra[i] == rank) {
        topoRanks->internalRank[c] = i;
        channel->ring2d.intra_prev = (i == 0) ? ring2DIntra[localRanks - 1] : ring2DIntra[i - 1];
        channel->ring2d.intra_next = (i == localRanks - 1) ? ring2DIntra[0] : ring2DIntra[i + 1];
      }
    }
    for (int i = 0; i < localRanks; i++) {
      int intraPeer = (topoRanks->internalRank[c] + i) % localRanks;
      intraRanks[(c + nChannels) * localRanks + i] = intraRanks[c * localRanks + i] = intraPeer;
    }
  }

  comm->algoEnable[NCCL_ALGO_RING] = 1;

  return ncclSuccess;
}

ncclResult_t ncclTopoRing2D::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  int nChannels = comm->nChannels, nRanks = comm->nRanks;
  int nNodes = comm->nNodes, localRanks = comm->localRanks;
  int node = comm->node, rank = comm->rank;
  if (nRanks != nNodes * localRanks)
    return ncclInvalidUsage;

  interRanks = new int[nNodes * MAXCHANNELS];

  for (int c = 0; c < nChannels; c++) {
    int localRank = allTopoRanks[comm->rank]->internalRank[c];
    struct ncclChannel *channel0 = comm->channels + c, *channel1 = channel0 + nChannels;

    for (int r = 0; r < nRanks; r++) {
      int r_node = allTopoRanks[r]->node;
      int r_localRank = allTopoRanks[r]->internalRank[c];
      if (r_localRank == localRank) {
        if (r_node == (node + 1) % nNodes)
          channel0->ring2d.inter_next = channel1->ring2d.inter_next = r;
        if ((r_node + 1) % nNodes == node)
          channel0->ring2d.inter_prev = channel1->ring2d.inter_prev = r;
      }
    }

    for (int i = 0; i < nNodes; i++) {
      int interPeer = (node + i) % nNodes;
      interRanks[(c + nChannels) * nNodes + i] = interRanks[c * nNodes + i] = interPeer;
    }

		channel0->ring2d.nIntraRanks = channel1->ring2d.nIntraRanks = localRanks;
		channel0->ring2d.nInterRanks = channel1->ring2d.nInterRanks = nNodes;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoRing2D::transportSetup() {
  int nNodes = comm->nNodes;
	char line[1024] = "";
	sprintf(line + strlen(line), "2D Ring for %d\n", comm->rank);
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;

		if (comm->localRanks > 1)
			NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->ring2d.intra_prev, 1, &channel->ring2d.intra_next));
    
    sprintf(line + strlen(line), "Intra Ranks: ");
    for (int i = 0; i < comm->localRanks; i++) {
      channel->ring2d.intraRanks[i] = intraRanks[c * comm->localRanks + i];
      sprintf(line + strlen(line), "%d/", channel->ring2d.intraRanks[i]);
    }
    sprintf(line + strlen(line), "\n");

    if (comm->nRanks == 1) continue;

    if (comm->nNodes > 1)
      NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->ring2d.inter_prev, 1, &channel->ring2d.inter_next));

    sprintf(line + strlen(line), "Inter Ranks: ");
    for (int i = 0; i < nNodes; i++) {
      channel->ring2d.interRanks[i] = interRanks[c * nNodes + i];
      sprintf(line + strlen(line), "%d/", channel->ring2d.interRanks[i]);
    }
    sprintf(line + strlen(line), "\n");

    sprintf(line + strlen(line), "nIntra: %d nPeer %d\n", channel->ring2d.nIntraRanks, channel->ring2d.nInterRanks);
		sprintf(line + strlen(line), "Ring: %d -> %d -> %d\n", channel->ring2d.intra_prev, comm->rank, channel->ring2d.intra_next);
    sprintf(line + strlen(line), "Ring: %d -> %d -> %d\n", channel->ring2d.inter_prev, comm->rank, channel->ring2d.inter_next);
  }
  delete[] interRanks;
  delete[] intraRanks;
  INFO(NCCL_COLL, "%s", line);
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternBroadcast;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternRing2D;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::enqueuePattern(struct ncclInfo *info, bool *redirect) const {
  if (info->coll == ncclCollBroadcast) {
    info->algorithm = NCCL_ALGO_RING;
    *redirect = true;
    return ncclSuccess;
  }
  NCCLCHECK(this->ncclEnqueueBase::enqueuePattern(info, redirect));
  return ncclSuccess;
}

int ncclEnqueueRing2D::getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, int nchunksPerLoop, int nstepsPerLoop) const {
  // Compute nSteps for proxies
  int stepSize = info->comm->buffSizes[info->protocol] / NCCL_STEPS;
  int chunkEffectiveSize = stepSize * args->chunkSteps;
  if (info->protocol == NCCL_PROTO_LL)
    chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128)
    chunkEffectiveSize = (chunkEffectiveSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  // if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d
  // (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels,
  // info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = 2 * (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels)) * nchunksPerLoop * chunkEffectiveSize)));
  return nstepsPerLoop * nLoops * args->chunkSteps;
}

ncclResult_t ncclEnqueueRing2D::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const {
  int pattern = info->pattern;
  struct ncclRing2D *ring2d = &args->channel->ring2d;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternRing2D) {
		if (info->comm->localRanks > 1) {
      NCCLCHECK(SaveProxy<proxySend>(ring2d->intra_next, args));
			NCCLCHECK(SaveProxy<proxyRecv>(ring2d->intra_prev, args));
		}
    if (info->comm->nNodes > 1) {
      int nsteps = getNsteps(args, info, ring2d->nInterRanks, ring2d->nInterRanks - 1);
      WARN("@@: %d", nsteps);
      NCCLCHECK(SaveProxy<proxySend>(ring2d->inter_next, args, nsteps));
      NCCLCHECK(SaveProxy<proxyRecv>(ring2d->inter_prev, args, nsteps));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueRing2D::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternRing2D:
    info->nchunksPerLoop = info->comm->localRanks;
    info->nstepsPerLoop = 2 * (info->comm->localRanks - 1);
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}
