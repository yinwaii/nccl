#ifndef __RING_REDUCE_SCATTER_H__
#define __RING_REDUCE_SCATTER_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
      prims(tid, args->nThreads, &ring->prev, &ring->next, NULL, channel, comm, 0);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->coll.lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128) {
        const ssize_t minChunkSize = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
        realChunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
      }
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*realChunkSize;

      /////////////// begin ReduceScatter steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ring->devUserRanks[nranks-1];
      offset = chunkOffset + rankDest * size;

      prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        prims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ring->devUserRanks[0];
      offset = chunkOffset + rankDest * size;

      prims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
    }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL128>(args);
    }
};

#endif