#ifndef __TREE_ALL_REDUCE_H__
#define __TREE_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-2*WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
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

#if 1
    if (tid < nthreads+WARP_SIZE) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, FUNC, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 0, ProtoSimple<1, 1, UNROLL>>
        prims(tid, args->nThreads, tree->down, &tree->up, NULL, channel, comm, 0);
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

    if (tid < nthreads+WARP_SIZE) {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, FUNC, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, ProtoSimple<1, 1, UNROLL>>
        prims(tid, args->nThreads, &tree->up, tree->down, thisOutput, channel, comm, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (tree->up == -1) {
          prims.directSend(thisOutput+offset, offset, nelem);
        } else if (tree->down[0] == -1) {
          prims.directRecv(thisOutput+offset, offset, nelem);
        } else {
          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }
      }
    }
#else
    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit == 256) nthreadsSplit += 64;
    if (tree->up == -1) {
      if (tid < nthreads+WARP_SIZE) {
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid, nthreads, tree->down, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
        }
      }
    } else {
      if (tid < nthreadsSplit + WARP_SIZE) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
          prims(tid, nthreadsSplit, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.send(thisInput+offset, nelem);
          } else {
            prims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &tree->up, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 2);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(thisOutput+offset, offset, nelem);
          } else {
            prims.directRecvCopySend(thisOutput+offset, offset, nelem);
          }
        }
      }
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
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

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      Primitives<T, FUNC, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 1, ProtoLL> LLprims(tid, nthreads, tree->down, &tree->up, NULL, channel, comm);
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
    } while(0);

    do {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      Primitives<T, FUNC, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, ProtoLL> LLprims(tid, nthreads, &tree->up, tree->down, NULL, channel, comm);
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
    } while(0);
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    const ssize_t loopSize = nChannels*chunkSize;
    int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (tree->up == -1) {
      // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
      Primitives<T, FUNC, FanAsymmetric<NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY>, 1, ProtoLL128> LLprims(tid, nthreads, tree->down, tree->down, NULL, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
      }
    } else {
      if (tid < nthreadsSplit) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        Primitives<T, FUNC, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 1, ProtoLL128> LLprims(tid, nthreadsSplit, tree->down, &tree->up, NULL, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.send(thisInput+offset, nelem);
          } else {
            LLprims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        Primitives<T, FUNC, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, ProtoLL128> LLprims(tid - nthreadsSplit, nthreads - nthreadsSplit, &tree->up, tree->down, NULL, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.recv(thisOutput+offset, nelem);
          } else {
            LLprims.recvCopySend(thisOutput+offset, nelem);
          }
        }
      }
    }
  }
};

#endif