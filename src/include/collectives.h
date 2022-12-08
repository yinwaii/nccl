/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "algo_config.h"

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#define FUNC_INDEX_P2P 0
#define FUNC_INDEX(coll, redop, dtype, al, pr) (1+(((((coll)*ncclNumOps + (redop))*ncclNumTypes) + (dtype))*NCCL_NUM_ALGORITHMS+(al))*NCCL_NUM_PROTOCOLS+(pr))

#define NCCL_COLL_NAME(coll, algo, proto, redop, dtype) \
  ncclFunction_##coll##_##algo##_##proto##_##redop##_##dtype

#define NCCL_KERN_NAME(coll, algo, proto, redop, dtype) \
  ncclKernel_##coll##_##algo##_##proto##_##redop##_##dtype

#define NCCL_IMPL_NAME(coll, algo, proto) \
  nccl##coll##algo##proto

#define WITH_COMMA(content) content,

/* Declare all collective operations */
#define DECL5(coll, algo, proto, redop, dtype) \
  extern __device__ void NCCL_COLL_NAME(coll, algo, proto, redop, dtype)(struct CollectiveArgs* args); \
  extern __global__ void NCCL_KERN_NAME(coll, algo, proto, redop, dtype)(struct ncclColl c); \

#define DECL4(coll, algo, redop, dtype) \
  DECL5(coll, algo, SIMPLE, redop, dtype) \
  DECL5(coll, algo, LL,     redop, dtype) \
  DECL5(coll, algo, LL128,  redop, dtype) 

#define DECL4_ELE(algo, coll, redop, dtype) \
  DECL4(coll, algo, redop, dtype)

#define DECL3(coll, redop, dtype) \
  MAP_FOR_ALGOS(DECL4_ELE, coll, redop, dtype)

#define DECL2(coll, redop) \
  DECL3(coll, redop, int8_t) \
  DECL3(coll, redop, uint8_t) \
  DECL3(coll, redop, int32_t) \
  DECL3(coll, redop, uint32_t) \
  DECL3(coll, redop, int64_t) \
  DECL3(coll, redop, uint64_t) \
  DECL3(coll, redop, half) \
  DECL3(coll, redop, float) \
  DECL3(coll, redop, double)

#define DECL(coll) \
  DECL2(coll, Sum) \
  DECL2(coll, Prod) \
  DECL2(coll, Min) \
  DECL2(coll, Max)

#define DECL_ALL \
  DECL2(Broadcast, Sum) \
  DECL(Reduce) \
  DECL2(AllGather, Sum) \
  DECL(ReduceScatter) \
  DECL(AllReduce) \
  DECL5(SendRecv, RING, SIMPLE, Sum, int8_t) \

DECL_ALL

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SENDRECV_SLICEFACTOR 4

#endif
