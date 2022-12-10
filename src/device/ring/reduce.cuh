/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __RING_REDUCE_H__
#define __RING_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runRing(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCE_CHUNKSTEPS : 1));
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;
    const int rank = ring->devUserRanks[0];
    const int prevRank = ring->devUserRanks[nranks-1];
    const int root = args->coll.root;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
      prims(tid, nthreads, &ring->prev, &ring->next, NULL, channel, comm);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      int realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->coll.lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128) {
        const ssize_t minChunkSize = int(nthreads * (Proto::calcBytePerGrain() / sizeof(T)));
        realChunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
      }
      ssize_t offset = gridOffset + bid*realChunkSize;
      int nelem = min(realChunkSize, size-offset);
      if (prevRank == root) {
        prims.send(thisInput+offset, nelem);
      } else if (rank == root) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct CollectiveArgs* args) {
      using Proto = ProtoSimple<REDUCE_CHUNKSTEPS / REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct CollectiveArgs* args) {
      runRing<T, RedOp, ProtoLL>(args);
    }
};

#include "prims_ll128.cuh"
template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct CollectiveArgs* args) {
      runRing<T, RedOp, ProtoLL128>(args);
    }
};

#endif