/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "collnets.h"
#include "algorithm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

typedef ncclResult_t (*ncclTopoPresetFunc_t)(struct ncclComm *comm, struct ncclTopoGraph *graph, struct ncclTopoRanks *topoRanks);
static const ncclTopoPresetFunc_t ncclTopoPresetFunc[NCCL_NUM_ALGORITHMS] = { ncclTopoPresetTree, ncclTopoPresetRing, ncclTopoPresetCollNet };

ncclResult_t ncclTopoPreset(struct ncclComm* comm, ncclAlgo **algos, struct ncclTopoRanks* topoRanks) {
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

typedef ncclResult_t (*ncclTopoPostsetFunc_t)(struct ncclComm *comm, struct ncclTopoGraph *graph, int *firstRanks, struct ncclTopoRanks **allTopoRanks);
static const ncclTopoPostsetFunc_t ncclTopoPostsetFunc[NCCL_NUM_ALGORITHMS] = {ncclTopoPostsetTree, ncclTopoPostsetRing, ncclTopoPostsetCollNet};

ncclResult_t ncclTopoPostset(struct ncclComm* comm, ncclAlgo** algos, int* firstRanks, struct ncclTopoRanks** allTopoRanks) {
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    // NCCLCHECK(algos[a]->topoPostset(firstRanks, allTopoRanks));
  NCCLCHECK(ncclTopoPostsetFunc[a](comm, &(algos[a]->graph), firstRanks, allTopoRanks));

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  int c;
  extern int *rings;
  for (c=nChannels; c<ncclMinNchannels(); c++) {
    // int *rings = dynamic_cast<ncclAlgoRing *>(algos[NCCL_ALGO_RING])->rings;
    memcpy(rings + c * nranks, rings + (c - nChannels) * nranks, nranks * sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-nChannels, sizeof(struct ncclChannel));
  }
  nChannels = comm->nChannels = c;

  return ncclSuccess;
}
