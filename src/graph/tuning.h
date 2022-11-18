#ifndef __TUNING_H__
#define __TUNING_H__
#include "comm.h"

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2

// LL128 max BW (per channel) for the different collectives
// ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce
const double ll128MaxBwPerCh[NCCL_NUM_FUNCTIONS] = { 18.8, 12.0, 18.3, 15.2, 16.7 };

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
static const float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][23] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .5,  .6,  .6,  .7,  .7,  .8,  .9,  .9, .92, .92 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .6,  .7,  .8,  .7,  .7,  .8,  .9,  .9 }
};

int getNthreads(const char *name, int env, int min, int max, int def);
int64_t ncclParamNthreads();
int64_t ncclParamLl128Nthreads();

#endif