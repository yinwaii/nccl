#ifndef __TREE_ALL_REDUCE_H__
#define __TREE_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runTreeUpDown(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    ssize_t chunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? args->coll.lastChunkSize
                   /* LL & LL128 */  : Proto::calcBytePerStep(comm)/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads-2*WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T))
                   /* LL & LL128 */  : nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;

    if (loopSize > size)
      chunkSize = DIVUP((int)size, int(nChannels*minChunkSize))*int(minChunkSize);

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;

    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 0, Proto>
      primsUp(tid, args->nThreads, tree->down, &tree->up, NULL, channel, comm, 0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1)
        primsUp.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      else if (tree->down[0] == -1)
        primsUp.send(thisInput+offset, nelem);
      else
        primsUp.recvReduceSend(thisInput+offset, nelem);
    }

    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, Proto>
      primsDn(tid, args->nThreads, &tree->up, tree->down, thisOutput, channel, comm, 0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1)
        primsDn.directSend(thisOutput+offset, offset, nelem);
      else if (tree->down[0] == -1)
        primsDn.directRecv(thisOutput+offset, offset, nelem);
      else
        primsDn.directRecvCopySend(thisOutput+offset, offset, nelem);
    }
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ void runTreeSplit(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    ssize_t chunkSize = int(
      Proto::Id != NCCL_PROTO_LL ? args->coll.lastChunkSize
                                 : Proto::calcBytePerStep(comm)/sizeof(T));
    const ssize_t minChunkSize = int(
      Proto::Id == NCCL_PROTO_SIMPLE ? (nthreads - 2*WARP_SIZE)*8*(sizeof(uint64_t)/sizeof(T)) :
      Proto::Id == NCCL_PROTO_LL     ? nthreads*(Proto::calcBytePerGrain()/sizeof(T))
                   /* LL128 */       : nthreads*(Proto::calcBytePerGrain()/sizeof(T))/8);
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;

    int nthreadsSplit;
    if (Proto::Id == NCCL_PROTO_SIMPLE) {
      nthreadsSplit = nthreads/2;
      if (nthreadsSplit >= 256) nthreadsSplit += 64;
    } else { // LL & LL128
      // Receiving from up to 3 sources is more compute intensive than sending
      // to 3 dests. Use 70% for reduce and 30% for bcast.
      nthreadsSplit = (nthreads*7/(10*WARP_SIZE))*WARP_SIZE;
    }

    if (loopSize > size)
      chunkSize = DIVUP((int)size, int(nChannels*minChunkSize))*int(minChunkSize);

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;

    if (tree->up == -1) {
      // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
      Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY>, 1, Proto> 
        prims(tid, nthreads, tree->down, tree->down, NULL, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
      }
    } else {
      if (tid < nthreadsSplit) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        // Why Direct=1????
        // Answer: Because despite not performing any direct operations, the ctor
        // must assume Direct so that it can exchange direct pointers with remote ctors
        // that are Direct, otherwise it hangs. A cleaner solution would be to seperate
        // into DirectRecv and DirectSend capabilities, this ctor would have both=0,
        // but the ctor above for tree roots would be DirectRecv=0 DirectSend=1.

        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DEV_ARITY, 1>, 1, Proto> 
          prims(tid, nthreadsSplit, tree->down, &tree->up, NULL, channel, comm, 0*Proto::MaxGroupWidth);
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
        Primitives<T, RedOp, FanAsymmetric<1, NCCL_MAX_DEV_ARITY>, 1, Proto>
          prims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, thisOutput, channel, comm, 1*Proto::MaxGroupWidth);
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
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runTreeUpDown<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runTreeSplit<T, RedOp, ProtoLL128>(args);
  }
};

#endif