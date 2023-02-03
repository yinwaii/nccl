#include "algo_interface.h"
#include <assert.h>

// Topo

ncclTopoButterfly2D::ncclTopoButterfly2D(struct ncclComm *comm): ncclTopoBase(NCCL_ALGO_BUTTERFLY_2D, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoButterfly2D::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  NCCLCHECK(ncclCalloc(&intraRanks, localRanks * MAXCHANNELS));

  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    int *butterfly2DIntra = graph.intra + c * localRanks;
    for (int i = 0; i < localRanks; i++) {
      if (butterfly2DIntra[i] == rank) {
        topoRanks->internalRank[c] = i;
        channel->butterfly2d.intra_prev = (i == 0) ? butterfly2DIntra[localRanks - 1] : butterfly2DIntra[i - 1];
        channel->butterfly2d.intra_next = (i == localRanks - 1) ? butterfly2DIntra[0] : butterfly2DIntra[i + 1];
      }
    }
    for (int i = 0; i < localRanks; i++) {
      int intraPeer = (topoRanks->internalRank[c] + i) % localRanks;
      intraRanks[(c + nChannels) * localRanks + i] = intraRanks[c * localRanks + i] = intraPeer;
    }
  }

  if (comm->algoEnable[NCCL_ALGO_BUTTERFLY_2D] == 1)
    comm->algoEnable[NCCL_ALGO_RING] = 1;

  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly2D::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  int nChannels = comm->nChannels, nRanks = comm->nRanks;
  int nNodes = comm->nNodes, localRanks = comm->localRanks;
  int node = comm->node, rank = comm->rank;
  if (nRanks != nNodes * localRanks)
    return ncclInvalidUsage;

  NCCLCHECK(ncclCalloc(&peerRanks, log2i(nNodes) * MAXCHANNELS));

  for (int c = 0; c < nChannels; c++) {
    int localRank = allTopoRanks[comm->rank]->internalRank[c];
    struct ncclChannel *channel0 = comm->channels + c, *channel1 = channel0 + nChannels;

    channel0->butterfly2d.edgeRank = channel1->butterfly2d.edgeRank = -1;
		for (int mask = 0; mask < log2i(nNodes); mask++) {
			peerRanks[(c+nChannels) * log2i(nNodes) + mask] = peerRanks[c * log2i(nNodes) + mask] = -1;
		}

    for (int r = 0; r < nRanks; r++) {
      int r_node = allTopoRanks[r]->node;
      int r_localRank = allTopoRanks[r]->internalRank[c];
      if (r_localRank == localRank) {
        int edge = 1 << log2i(nNodes), edgePeer = node ^ edge;
        if (r_node == edgePeer)
          channel0->butterfly2d.edgeRank = channel1->butterfly2d.edgeRank = r;
        for (int mask = 0; mask < log2i(nNodes); mask++) {
          int peer = node ^ (1 << mask);
					if ((node & edge) == 0 && r_node == peer)
						peerRanks[(c+nChannels) * log2i(nNodes) + mask] = peerRanks[c * log2i(nNodes) + mask] = r;
        }
      }
    }

		channel0->butterfly2d.nIntraRanks = channel1->butterfly2d.nIntraRanks = localRanks;
		channel0->butterfly2d.nPeerRanks = channel1->butterfly2d.nPeerRanks = log2i(nNodes);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly2D::topoDuplicate(int c) {
  memcpy(intraRanks + c * comm->localRanks, intraRanks + (c-comm->nChannels) * comm->localRanks, comm->localRanks * sizeof(int));
  memcpy(peerRanks + c * log2i(comm->nNodes), peerRanks + (c-comm->nChannels)*log2i(comm->nNodes), log2i(comm->nNodes) * sizeof(int));
  return ncclSuccess;
}

ncclResult_t ncclTopoButterfly2D::transportSetup() {
  int nNodes = comm->nNodes;

  for (int c = 0; c < comm->nChannels; c++) {
    char line[1024] = "";
	  sprintf(line + strlen(line), "2D Butterfly for %d %d\n", comm->rank, c);
    
    struct ncclChannel *channel = comm->channels + c;

		if (comm->localRanks > 1)
			NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->butterfly2d.intra_prev, 1, &channel->butterfly2d.intra_next));
    
    sprintf(line + strlen(line), "Intra Ranks: ");
    for (int i = 0; i < comm->localRanks; i++) {
      channel->butterfly2d.intraRanks[i] = intraRanks[c * comm->localRanks + i];
      sprintf(line + strlen(line), "%d/", channel->butterfly2d.intraRanks[i]);
    }
    sprintf(line + strlen(line), "\n");

    int edgePeer = channel->butterfly2d.edgeRank;
    sprintf(line + strlen(line), "Channel %d: edgeRanks %d\n", c, channel->butterfly2d.edgeRank);
    if (edgePeer != -1)
      NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &edgePeer, 1, &edgePeer));

    if (comm->nRanks == 1) continue;
    sprintf(line + strlen(line), "Peer Ranks: ");
    for (int i = 0; i < log2i(nNodes); i++) {
      channel->butterfly2d.peerRanks[i] = peerRanks[c * log2i(nNodes) + i];
      int peer = channel->butterfly2d.peerRanks[i];
      sprintf(line + strlen(line), "%d/", peer);
      if (peer != -1)
        NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &peer, 1, &peer));
    }
    sprintf(line + strlen(line), "\n");
    sprintf(line + strlen(line), "nIntra: %d nPeer %d\n", channel->butterfly2d.nIntraRanks, channel->butterfly2d.nPeerRanks);
		sprintf(line + strlen(line), "Ring: %d -> %d -> %d\n", channel->butterfly2d.intra_prev, comm->rank, channel->butterfly2d.intra_next);

    INFO(NCCL_COLL, "%s", line);
  }
  free(peerRanks);
  free(intraRanks);
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly2D::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternBroadcast;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternButterfly2D;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly2D::enqueueRedirect(struct ncclInfo *info) const {
  if (info->coll == ncclCollBroadcast) {
    info->comm->algoEnable[NCCL_ALGO_RING] = 1;
    info->comm->algoEnable[NCCL_ALGO_BUTTERFLY_2D] = 0;
  }
  return ncclSuccess;
}

int ncclEnqueueButterfly2D::getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, size_t size, int nstepsPerLoop) const {
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
  int nLoops = (int)(DIVUP(size, (((size_t)(info->nChannels)) * info->nchunksPerLoop * chunkEffectiveSize)));
  return nstepsPerLoop * nLoops * args->chunkSteps;
}

ncclResult_t ncclEnqueueButterfly2D::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const {
  int pattern = info->pattern;
  struct ncclButterfly2D *butterfly2d = &args->channel->butterfly2d;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternButterfly2D) {
		if (info->comm->localRanks > 1) {
      NCCLCHECK(SaveProxy<proxySend>(butterfly2d->intra_next, args));
			NCCLCHECK(SaveProxy<proxyRecv>(butterfly2d->intra_prev, args));
		}
		if (butterfly2d->edgeRank != -1) {
      int nsteps = getNsteps(args, info, info->nBytes, 1);
			NCCLCHECK(SaveProxy<proxySend>(butterfly2d->edgeRank, args, nsteps));
			NCCLCHECK(SaveProxy<proxyRecv>(butterfly2d->edgeRank, args, nsteps));
		}
		for (int i = 0; i < log2i(info->comm->nNodes); i++) {
			int peer = butterfly2d->peerRanks[i];
			int nsteps = 2 * getNsteps(args, info, (info->nBytes >> i) / 2, 1);
			NCCLCHECK(SaveProxy<proxySend>(peer, args, nsteps));
      NCCLCHECK(SaveProxy<proxyRecv>(peer, args, nsteps));
		}
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueButterfly2D::enqueueLoopInfo(struct ncclInfo *info) const {
  switch (info->pattern) {
  case ncclPatternButterfly2D:
    info->nchunksPerLoop = info->comm->localRanks;
    info->nstepsPerLoop = info->comm->localRanks - 1;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}
