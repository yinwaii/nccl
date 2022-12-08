/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "common.cuh"
#include "algo_config.h"

__device__ volatile uint64_t* ncclShmem;

#define NCCL_FUNC5(coll, algo, redop, type)                   \
  WITH_COMMA(NCCL_COLL_NAME(coll, algo, LL,     redop, type)) \
  WITH_COMMA(NCCL_COLL_NAME(coll, algo, LL128,  redop, type)) \
  WITH_COMMA(NCCL_COLL_NAME(coll, algo, SIMPLE, redop, type))

#define NCCL_FUNC5_ELE(algo, coll, redop, type) \
  NCCL_FUNC5(coll, algo, redop, type)

#define NCCL_FUNC4(coll, redop, type) \
  MAP_FOR_ALGOS(NCCL_FUNC5_ELE, coll, redop, type)

// Must be consistent with ncclDataType_t
#define NCCL_FUNCS3A(coll, redop) \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, uint8_t)  \
  NCCL_FUNC4(coll, redop, int32_t)  \
  NCCL_FUNC4(coll, redop, uint32_t)  \
  NCCL_FUNC4(coll, redop, int64_t)  \
  NCCL_FUNC4(coll, redop, uint64_t)  \
  NCCL_FUNC4(coll, redop, half)  \
  NCCL_FUNC4(coll, redop, float)  \
  NCCL_FUNC4(coll, redop, double)
#define NCCL_FUNCS3B(coll, redop) \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)  \
  NCCL_FUNC4(coll, redop, int8_t)

// Must be consistent with ncclRedOp_t
#define NCCL_FUNCS2A(coll) \
  NCCL_FUNCS3A(coll, Sum )  \
  NCCL_FUNCS3A(coll, Prod)  \
  NCCL_FUNCS3A(coll, Max )  \
  NCCL_FUNCS3A(coll, Min )
#define NCCL_FUNCS2B(coll) \
  NCCL_FUNCS3B(coll, Sum)  \
  NCCL_FUNCS3B(coll, Sum)  \
  NCCL_FUNCS3B(coll, Sum)  \
  NCCL_FUNCS3B(coll, Sum)

// Must be consistent with ncclFunc_t
#define NCCL_FUNCS()  \
  WITH_COMMA(NCCL_COLL_NAME(SendRecv, RING, SIMPLE, Sum, int8_t))\
  NCCL_FUNCS2B(Broadcast) \
  NCCL_FUNCS2A(Reduce) \
  NCCL_FUNCS2B(AllGather) \
  NCCL_FUNCS2A(ReduceScatter) \
  NCCL_FUNCS2A(AllReduce) 

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ncclFuncs[1+NCCL_NUM_FUNCTIONS*ncclNumOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  NCCL_FUNCS()
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ncclWorkaroundClangD55580() {}
