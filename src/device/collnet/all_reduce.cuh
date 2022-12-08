/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      struct ncclTree* tree = &channel->collTreeUp;
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.send(thisInput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      struct ncclTree* tree = &channel->collTreeDn;
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.recv(thisOutput+offset, nelem);
        } else {
          prims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      struct ncclTree* tree = &channel->collTreeUp;
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }

    if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
      struct ncclTree* tree = &channel->collTreeDn;
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          LLprims.send(thisOutput+offset, nelem);
        } else if (tree->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
};
