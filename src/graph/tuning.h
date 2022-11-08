#ifndef __TUNING_H__
#define __TUNING_H__
#include "comm.h"

// Latencies in us, Bandwidths in GB/s
// Tree { LL, LL128, Simple } , Ring { LL, LL128, Simple }
const float baseLat  [NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = { { 4.4, 4.4,  0 }, { 3.6, 10.0, 8.4 }, { 4.4, 4.4,  0 } };

// NVLink, PCI, Network
#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
// Tree/Simple is the latency a 256kB chunk, which is ~ base lat + 256k/12GB/s (+ 256k/12GB/s for the network).
const float hwLat [3][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] =
{ /* NVLINK */
  { /* Tree (LL/LL128/Simple)*/ { .52, 1.2, 28 }, /* Ring (LL/LL128/Simple)*/ { .47, 1.9, 3.4 }, /* CollNet (LL/LL128/Simple)*/ {  .5, 1.2, 4.0 } },
  /* PCI */
  { /* Tree (LL/LL128/Simple)*/ { 1.0, 1.9, 28 }, /* Ring (LL/LL128/Simple)*/ { 1.0, 2.5, 5.7 }, /* CollNet (LL/LL128/Simple)*/ { 1.0, 1.9, 5.5 } },
  /* NET */
  { /* Tree (LL/LL128/Simple)*/ { 5.0, 8.5, 50 }, /* Ring (LL/LL128/Simple)*/ { 2.7, 4.0, 9.6 }, /* CollNet (LL/LL128/Simple)*/ { 5.0, 5.0, 10.7 } }
};

// LL128 max BW (per channel) for the different collectives
// ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce
const double ll128MaxBwPerCh[NCCL_NUM_FUNCTIONS] = { 18.8, 12.0, 18.3, 15.2, 16.7 };

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 128MB.
const float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][22] = {
  { 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .7,  .7,  .7,  .7,  .6,  .5,  .4,  .4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.0, 1.0 },
  { 1.0, 1.0, 1.0, 1.0, 1.0,  .9,  .8,  .8,  .8,  .7,  .6,  .6,  .6,  .5,  .6,  .6,  .7,  .7,  .8,  .9,  .9, 1.0 },
  {  .9,  .9,  .9,  .9,  .9,  .9,  .9,  .8,  .7,  .6,  .6,  .5,  .5,  .5,  .5,  .5,  .5,  .6,  .6,  .7,  .8,  .9 }
};

int getNthreads(const char *name, int env, int min, int max, int def);
int64_t ncclParamNthreads();
int64_t ncclParamLl128Nthreads();
ncclResult_t ncclTuningThresholds(struct ncclComm *comm, int a);

#endif