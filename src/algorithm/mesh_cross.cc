#include "algo_interface.h"
#include <assert.h>
#include <math.h>

// Topo

ncclTopoMeshCross::ncclTopoMeshCross(struct ncclComm *comm)
    : ncclTopoBase(NCCL_ALGO_MESH_CROSS, comm, ncclParamCrossNic(), 0) {}

ncclResult_t ncclTopoMeshCross::topoPreset(struct ncclTopoRanks *topoRanks) {
  int rank = comm->rank, nRanks = comm->nRanks;
  int nPartitions = 1;
  for (int i = 1; i * i <= nRanks; i++) {
    if (nRanks % (i * i) == 0 && i > nPartitions)
      nPartitions = i;
  }
  int localRanks = nRanks / nPartitions, nSubRanks = localRanks / nPartitions;
  int nChannels = comm->nChannels;
  comm->nPartitions = nPartitions;
  comm->nSubRanks = nSubRanks;
  // printf("nPart %d, localR %d, nSubChannels %d", nPartitions, localRanks,
  // nSubRanks);

  for (int c = 0; c < nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    channel->meshCross.inter_prev = ((rank % localRanks) < nSubRanks) ? (rank + localRanks - nSubRanks) : (rank - nSubRanks);
		channel->meshCross.inter_next = ((rank % localRanks) + nSubRanks >= localRanks) ? (rank + nSubRanks - localRanks) : (rank + nSubRanks);
		channel->meshCross.intra_prev = (rank % localRanks == 0) ? (rank + localRanks - 1) : (rank - 1);
		channel->meshCross.intra_next = ((rank + 1) % localRanks == 0) ? (rank + 1 - localRanks) : (rank + 1);
		int row = rank / localRanks, col = (rank / nSubRanks) % nPartitions;
		channel->meshCross.mirror = col * localRanks + row * nSubRanks + (rank % nSubRanks);
		TRACE(NCCL_GRAPH, "Mesh %d(%d,%d)-%d: %d(up) -> %d(left) -> %d(mirror) -> %d(right) -> %d(down)", comm->rank, row, col, c, channel->meshCross.inter_prev, channel->meshCross.intra_prev, channel->meshCross.mirror, channel->meshCross.intra_next, channel->meshCross.inter_next);
  }

  if (comm->algoEnable[NCCL_ALGO_MESH_CROSS] == 1)
    comm->algoEnable[NCCL_ALGO_BUTTERFLY_YZ] = 1;
  return ncclSuccess;
}

ncclResult_t ncclTopoMeshCross::topoPostset(int *firstRanks, struct ncclTopoRanks **allTopoRanks) {
  return ncclSuccess;
}

ncclResult_t ncclTopoMeshCross::transportSetup() {
  int nSubRanks = comm->nSubRanks, localRanks = comm->nSubRanks * comm->nPartitions;
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel *channel = comm->channels + c;
    char line[1000];
    channel->meshCross.interRanks[0] = comm->rank;
    channel->meshCross.nInterRanks = comm->nPartitions;
    sprintf(line, "%d InterRanks: %d", comm->rank, channel->meshCross.interRanks[0]);
    for (int r = 1; r < comm->nPartitions; r++) {
      int prev = channel->meshCross.interRanks[r - 1];
      channel->meshCross.interRanks[r] = ((prev % localRanks) + nSubRanks >= localRanks) ? (prev + nSubRanks - localRanks) : (prev + nSubRanks);
      sprintf(line + strlen(line), " %d", channel->meshCross.interRanks[r]);
    }
    TRACE(NCCL_GRAPH, "%s", line);
    channel->meshCross.intraRanks[0] = comm->rank;
    channel->meshCross.nIntraRanks = localRanks;
    sprintf(line, "%d IntraRanks: %d", comm->rank, channel->meshCross.intraRanks[0]);
    for (int r = 1; r < localRanks; r++) {
      int prev = channel->meshCross.intraRanks[r - 1];
      channel->meshCross.intraRanks[r] = ((prev + 1) % localRanks == 0) ? (prev + 1 - localRanks) : (prev + 1);
      sprintf(line + strlen(line), " %d", channel->meshCross.intraRanks[r]);
    }
    TRACE(NCCL_GRAPH, "%s", line);
    if (comm->nRanks == 1)
      continue;
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.inter_prev, 1, &channel->meshCross.inter_next));
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.intra_prev, 1, &channel->meshCross.intra_next));
    NCCLCHECK(ncclTransportP2pSetup(comm, &graph, channel, 1, &channel->meshCross.mirror, 1, &channel->meshCross.mirror));
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::getPattern(int coll, int *pattern) const {
  switch (coll) {
  case ncclCollBroadcast:
    *pattern = ncclPatternBroadcast;
    break;
  case ncclCollAllReduce:
    *pattern = ncclPatternMeshCross;
    break;
  default:
    *pattern = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::enqueueRedirect(struct ncclInfo *info) const {
  if (info->coll == ncclCollBroadcast) {
    info->comm->algoEnable[NCCL_ALGO_BUTTERFLY_YZ] = 1;
    info->comm->algoEnable[NCCL_ALGO_MESH_CROSS] = 0;
  }
  return ncclSuccess;
}

int ncclEnqueueMeshCross::getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info, int nstepsPerLoop) const {
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
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels)) * info->nchunksPerLoop * chunkEffectiveSize)));
  return nstepsPerLoop * nLoops * args->chunkSteps;
}

ncclResult_t ncclEnqueueMeshCross::proxySaveColl(struct ncclProxyArgs *args, struct ncclInfo *info) const {
  int pattern = info->pattern;
  struct ncclMeshCross *meshCross = &args->channel->meshCross;
  int nRanks = info->comm->nRanks;
  if (pattern == ncclPatternMeshCross) {
    int nInterSteps = getNsteps(args, info, 2 * (info->comm->nPartitions - 1));
    int nIntraSteps = args->nsteps;
    // printf("nInterSteps is %ld\n", nInterSteps);
    if (meshCross->nInterRanks != meshCross->nIntraRanks && nInterSteps > 0) {
      NCCLCHECK(SaveProxy<proxyRecv>(meshCross->inter_prev, args, nInterSteps));
      NCCLCHECK(SaveProxy<proxySend>(meshCross->inter_next, args, nInterSteps));
    }
    NCCLCHECK(SaveProxy<proxyRecv>(meshCross->intra_prev, args, nIntraSteps + (meshCross->nInterRanks == meshCross->nIntraRanks ? nInterSteps : 0)));
    NCCLCHECK(SaveProxy<proxySend>(meshCross->intra_next, args, nIntraSteps + (meshCross->nInterRanks == meshCross->nIntraRanks ? nInterSteps : 0)));
    if (info->comm->rank != meshCross->mirror) {
      int nSteps = getNsteps(args, info, 2 * info->comm->nPartitions);
      NCCLCHECK(SaveProxy<proxySend>(meshCross->mirror, args, nSteps));
      NCCLCHECK(SaveProxy<proxyRecv>(meshCross->mirror, args, nSteps));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclEnqueueMeshCross::enqueueLoopInfo(struct ncclInfo *info) const {
  int localRanks = info->comm->nPartitions * info->comm->nSubRanks;
  switch (info->pattern) {
  case ncclPatternMeshCross:
    info->nchunksPerLoop = localRanks * info->comm->nPartitions;
    info->nstepsPerLoop = 2 * (localRanks - 1) * info->comm->nPartitions;
    break;
  default:
    WARN("Unknown pattern %d\n", info->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}
