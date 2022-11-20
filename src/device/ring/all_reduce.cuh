#ifndef __RING_ALL_REDUCE_H__
#define __RING_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
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
    const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
      prims(tid, args->nThreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
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
    const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(thisOutput+offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(thisOutput+offset, nelem);
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
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
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize;
    const ssize_t size = args->coll.count;

    ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSend(thisInput+offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvCopySend(thisOutput+offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      LLprims.recv(thisOutput+offset, nelem);
    }
  }
};

#endif