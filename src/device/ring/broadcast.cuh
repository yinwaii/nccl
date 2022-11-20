#ifndef __RING_BROADCAST_H__
#define __RING_BROADCAST_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * BROADCAST_CHUNKSTEPS;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      Primitives<T, FUNC, FanAsymmetric<1, 1>, 0, ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS, UNROLL>>
        prims(tid, args->nThreads, &ring->prev, &ring->next, NULL, channel, comm, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
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
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      Primitives<T, FUNC, FanAsymmetric<1, 1>, 1, ProtoLL> LLprims(tid, nthreads, &ring->prev, &ring->next, NULL, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (rank == root) {
          if (thisInput == thisOutput) {
            LLprims.send(thisInput+offset, nelem);
          } else {
            LLprims.copySend(thisInput + offset, thisOutput + offset, nelem);
          }
        } else if (nextRank == root) {
          LLprims.recv(thisOutput + offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput + offset, nelem);
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int nextRank = ring->devUserRanks[1];
      const int root = args->coll.root;

      Primitives<T, FUNC, FanAsymmetric<1, 1>, 1, ProtoLL128> LLprims(tid, nthreads, &ring->prev, &ring->next, NULL, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (rank == root) {
          if (thisInput == thisOutput) {
            LLprims.send(thisInput+offset, nelem);
          } else {
            LLprims.copySend(thisInput + offset, thisOutput + offset, nelem);
          }
        } else if (nextRank == root) {
          LLprims.recv(thisOutput + offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput + offset, nelem);
        }
      }
    }
};

#endif