#ifndef __RING_BROADCAST_H__
#define __RING_BROADCAST_H__
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
    const ssize_t chunkSize = int(Proto::calcBytePerStep(comm) / sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? BROADCAST_CHUNKSTEPS : 1));
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;
    const int rank = ring->devUserRanks[0];
    const int nextRank = ring->devUserRanks[1];
    const int root = args->coll.root;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
      prims(tid, args->nThreads, &ring->prev, &ring->next, NULL, channel, comm, 0);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->coll.lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128) {
        const ssize_t minChunkSize = int(nthreads * (Proto::calcBytePerGrain() / sizeof(T)));
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSize)*minChunkSize);
      }
      realChunkSize = int(realChunkSize);

      ssize_t offset = gridOffset + bid*realChunkSize;
      int nelem = min(realChunkSize, size-offset);

      if (rank == root) {
        if (thisInput == thisOutput) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.copySend(thisInput+offset, thisOutput+offset, nelem);
        }
      } else if (nextRank == root) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS / BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, UNROLL>;
      runRing<T, RedOp, Proto>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL>(args);
    }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      runRing<T, RedOp, ProtoLL128>(args);
    }
};

#endif