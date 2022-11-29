#ifndef __BUTTERFLY2_ALL_REDUCE_H__
#define __BUTTERFLY2_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
//   __host__ __device__ static long log2i(long n) {
//     long l = 0;
//     while (n >>= 1) l++;
//     return l;
//   }
  template<typename T, typename RedOp, typename Proto>
  __device__ void runButterfly2(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclButterfly* butterfly = &channel->butterfly;
    ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const ssize_t minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;
    int rank = comm->rank, commOffset = 0;

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;
    if (tid == 0) {
      for (int p = 0; p < log2i(comm->nRanks); p++) {
        int peer = butterfly->devPeerRanks[p];
        if (peer != -1) {
          printf("%d: Peer is %d\n", p, peer);
        }
      }
    }

    auto getRealChunkSize = [&] __device__(ssize_t gridOffset, ssize_t tailOffset) -> ssize_t {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(tailOffset-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, divUp(tailOffset-gridOffset, nChannels*minChunkSize)*minChunkSize);
      return int(realChunkSize);
    };

    auto reduce = [&]__device__(int peer, int step, bool scatter, bool edge) -> void {
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
        prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm, step * Proto::MaxGroupWidth);

      if (tid == 0)
        printf("%d: START FOR peer %d\n", comm->rank, peer);
      ssize_t length = edge ? size : (size >> (step + 1));
      ssize_t splitOffset = edge ? 0 : length;
      bool forward = rank < peer, direction = scatter ? forward : !forward;
      if (!forward && !scatter)
        commOffset -= splitOffset;
      ssize_t initOffset = edge ? 0 : commOffset;
      for (ssize_t gridOffset = initOffset; gridOffset < initOffset + length; gridOffset += loopSize)
      {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, initOffset + length);
        ssize_t chunkOffset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, initOffset + length - chunkOffset);
        if (tid == 0) {
          printf("%d: chunkOffset: %ld nelem: %d", comm->rank, chunkOffset, nelem);
        }
        if (!edge || edge && direction)
          prims.send(thisInput+chunkOffset+(direction?splitOffset:0), nelem);
        if (!edge || edge && !direction) {
          if (scatter)
            prims.recvReduceCopy(thisInput+chunkOffset+(!direction?splitOffset:0), thisOutput+chunkOffset+(!direction?splitOffset:0), nelem);
          else
            prims.recv(thisOutput+chunkOffset+(!direction?splitOffset:0), nelem);
        }
      }
      if (!forward && scatter)
        commOffset += splitOffset;
      if (tid == 0)
        printf("%d: COMPLETED FOR peer %d\n", comm->rank, peer);
    };

    int edgeRank = butterfly->edgeRank, nSteps = log2i(comm->nRanks);
    if (edgeRank != -1)
      reduce(edgeRank, nSteps, true, true);
    for (int p = 0; p < nSteps; p++) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        reduce(peer, p, true, false);
      }
    }
    for (int p = nSteps - 1; p >= 0; p--) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        reduce(peer, p, false, false);
      }
    }
    if (edgeRank != -1)
      reduce(edgeRank, nSteps, false, true);
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY2, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
	using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runButterfly2<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY2, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runButterfly2<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY2, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runButterfly2<T, RedOp, ProtoLL128>(args);
  }
};

#endif