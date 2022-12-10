/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COLLNET_ALL_REDUCE_H__
#define __COLLNET_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runCollNet(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    int chunkSize = (Proto::Id == NCCL_PROTO_SIMPLE ? args->coll.lastChunkSize : (Proto::calcBytePerStep(comm) / sizeof(T)));
    const ssize_t minChunkSize = (Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads * 8) : nthreads)* sizeof(uint64_t) / sizeof(T);
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
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto>
        prims(tid, nthreads, tree->down, &tree->up, NULL, channel, comm);
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
      Primitives<T, RedOp, FanAsymmetric<1, 1>, 0, Proto>
        prims(tid, nthreads, &tree->up, tree->down, NULL, channel, comm);
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
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runCollNet<T, RedOp, ProtoSimple<1, 1, UNROLL>>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runCollNet<T, RedOp, ProtoLL>(args);
  }
};
#endif
