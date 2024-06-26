/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "algo_interface.h"
#include "graph.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm, AlgoInfo<ncclTopoAlgo> algos, struct ncclTopoRanks* topoRanks) {
  int nChannels = comm->nChannels;

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    NCCLCHECK(algos[a]->topoPreset(topoRanks));

  // Duplicate channels rings/trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    WARN("User asked for a minimum of %d channels, limiting to %d\n", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1\n", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, AlgoInfo<ncclTopoAlgo> algos, int* firstRanks, struct ncclTopoRanks** allTopoRanks) {
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    // WARN("xxx %s: %d", ncclAlgoStr[a], comm->algoEnable[a]);
    if (comm->algoEnable[a])
      NCCLCHECK(algos[a]->topoPostset(firstRanks, allTopoRanks));
  }

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  int c;
  for (c=nChannels; c<ncclMinNchannels(); c++) {
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      if (comm->algoEnable[a])
        NCCLCHECK(algos[a]->topoDuplicate(c)); 
    }
    // int *rings = dynamic_cast<ncclTopoRing *>(algos[NCCL_ALGO_RING].get())->rings; 
    // if (rings == nullptr)
    //   return ncclInternalError;
    // memcpy(rings + c * nranks, rings + (c - nChannels) * nranks, nranks * sizeof(int));
    memcpy(comm->channels + c, comm->channels + c - nChannels, sizeof(struct ncclChannel));
  }
  nChannels = comm->nChannels = c;

  return ncclSuccess;
}
