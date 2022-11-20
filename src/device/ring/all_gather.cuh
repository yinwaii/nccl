#ifndef __RING_ALL_GATHER_H__
#define __RING_ALL_GATHER_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template <typename T, typename RedOp, typename Proto>
  __device__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm *comm = args->comm;
    struct ncclChannel *channel = comm->channels + blockIdx.x;
    struct ncclRing *ring = &channel->ring;
    const ssize_t chunkSize = int(Proto::calcBytePerStep(comm) / sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels * int(chunkSize);
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, channel, comm, 0);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(size - gridOffset, nChannels));
        ALIGN_SIZE(realChunkSize, (nthreads - WARP_SIZE) * sizeof(uint64_t) / sizeof(T));
      }
      if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size - gridOffset < loopSize ? args->coll.lastChunkSize : chunkSize;
      if (Proto::Id == NCCL_PROTO_LL128) {
          const ssize_t minChunkSize = int(nthreads * (Proto::calcBytePerGrain() / sizeof(T)) / 2);
          // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
          realChunkSize = min(chunkSize, DIVUP(size - gridOffset, nChannels * minChunkSize) * minChunkSize);
      }
      realChunkSize = int(realChunkSize);
      ssize_t chunkOffset = gridOffset + bid*realChunkSize;

      /////////////// begin AllGather steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ring->devUserRanks[0];
      offset = chunkOffset + rankDest * size;

      if (thisInput + chunkOffset == thisOutput + offset)  // In place
        prims.directSend(thisInput+chunkOffset, offset, nelem);
      else 
        prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      rankDest = ring->devUserRanks[1];
      offset = chunkOffset + rankDest * size;

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS / ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL128>(args);
    }
};

#endif