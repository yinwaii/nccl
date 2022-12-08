/*************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include "collectives.h"
#include "devcomm.h"

// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 13, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}

typedef void(*ncclKern_t)(struct CollectiveArgs* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}
static __device__ void load_coll(struct ncclColl* localColl, struct ncclColl* hostColl, int tid, struct ncclDevComm* comm) {
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort);
  load_parallel(localColl, hostColl, sizeof(struct ncclColl), tid);
  __syncthreads();
  if (tid == 0) hostColl->active = 0;
}

extern __device__ volatile uint64_t* ncclShmem;
template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction {
  public:
  __device__ void run(struct CollectiveArgs* args) {}
};

template <ncclFunc_t FUNCTION, int ALGO, int PROTO, class REDOP, typename T, int UNROLL, int FINDEX>
__device__ void ncclKernel(struct ncclColl firstColl)  {
// __global__ void NCCL_KERN_NAME(coll, op, dtype)(struct ncclColl firstColl) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ volatile uint64_t shmem[NCCL_LL128_SHMEM_SIZE];
  ncclShmem = shmem;
  __shared__ struct ncclColl localColl;
  auto f = ncclFunction<FUNCTION, ALGO, PROTO, REDOP, T, UNROLL>();
 
  struct ncclDevComm* comm = firstColl.args.comm;
  struct ncclChannel* channel = comm->channels+bid;
  struct ncclColl* c;
  if (bid == 0) {
    /* To optimize for latency, (only) the first operation is passed as argument.*/
    c = &firstColl;
  } else {
    c = &localColl;
    load_coll(c, channel->collectives+channel->collFifoHead, tid, comm);
  }
  while (1) {
    if (tid < c->args.common.nThreads) {
      if (c->funcIndex == FINDEX) {
        f.run(&c->args);
      } else {
        ncclFuncs[c->funcIndex](&c->args);
      }
    }
    int nextIndex = c->nextIndex;
    if (tid == 0) channel->collFifoHead = nextIndex;
 
    if (c->active == 2) {
      return;
    }

    /* Load next collective operation*/
    c = &localColl; /* for bid 0 */
    load_coll(c, channel->collectives+nextIndex, tid, comm);
  }
}

/* Functions for aggregation case */
#define IMPL_COLL_FUNC(coll, algo, proto, redop, type) \
__device__ void NCCL_COLL_NAME(coll, algo, proto, redop, type)(struct CollectiveArgs* args) { \
  auto f = ncclFunction<ncclColl##coll, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL>(); \
  f.run(args); \
}

#if NCCL_OP == 0
/* Kernels with the first operation inlined */
#define IMPL_COLL_KERN(coll, algo, proto, redop, type, fIndex) \
__global__ void NCCL_KERN_NAME(coll, algo, proto, redop, type)(struct ncclColl firstColl) { \
  ncclKernel<ncclColl##coll, NCCL_ALGO_##algo, NCCL_PROTO_##proto, Func##redop<type>, type, COLL_UNROLL, fIndex>(firstColl); \
}
#else
#define IMPL_COLL_KERN(coll, algo, proto, redop, type, fIndex)
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(coll, algo, redop, type, ncclType) \
  IMPL_COLL_FUNC(coll, algo, LL,     redop, type) \
  IMPL_COLL_FUNC(coll, algo, LL128,  redop, type) \
  IMPL_COLL_FUNC(coll, algo, SIMPLE, redop, type) \
  IMPL_COLL_KERN(coll, algo, LL,     redop, type, FUNC_INDEX(ncclColl##coll, nccl##redop, ncclType, NCCL_ALGO_##algo, NCCL_PROTO_LL)) \

#define IMPL_COLL4_ELE(algo, coll, redop, type, ncclType) \
  IMPL_COLL4(coll, algo,    redop, type, ncclType)

#define IMPL_COLL3(coll, redop, type, ncclType) \
  MAP_FOR_ALGOS(IMPL_COLL4_ELE, coll, redop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(coll, redop) IMPL_COLL3(coll, redop, double,   ncclFloat64)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(coll) IMPL_COLL2(coll, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(coll) IMPL_COLL2(coll, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(coll) IMPL_COLL2(coll, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(coll) IMPL_COLL2(coll, Max);
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(coll) IMPL_COLL3(coll, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(coll) \
  IMPL_COLL_FUNC(coll, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(coll, RING, SIMPLE, Sum, int8_t, 0);
#else
#define IMPL_COLL_C(coll)
#define IMPL_COLL_P(coll)
#endif

#define COLL_UNROLL 4

#endif
