#ifndef __BUTTERFLY_ALL_REDUCE_H__
#define __BUTTERFLY_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  __host__ __device__ static long log2i(long n) {
    long l = 0;
    while (n >>= 1) l++;
    return l;
  }
  template<typename T, typename RedOp, typename Proto>
  __device__ void runButterfly(struct CollectiveArgs* args) {
		const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclButterfly* butterfly = &channel->butterfly;
    ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const ssize_t minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    const ssize_t loopSize = int(nChannels*chunkSize);
    const ssize_t size = args->coll.count;

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

    auto reduce = [&]__device__(int peer, bool send, bool recv, int step)->void {
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
        prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm);

      if (tid == 0)
        printf("%d: START FOR peer %d\n", comm->rank, peer);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize;
        if (Proto::Id == NCCL_PROTO_SIMPLE) {
          realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
          ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        }
        else
          realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize);
        realChunkSize = int(realChunkSize);

        ssize_t chunkOffset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, size - chunkOffset);
        if (tid == 0) {
          printf("%d: chunkOffset: %ld nelem: %d\n", comm->rank, chunkOffset, nelem);
        }
        if (send)
          prims.send((step>0?thisOutput:thisInput)+chunkOffset, nelem);
        if (recv)
          prims.recvReduceCopy((step>0?thisOutput:thisInput)+chunkOffset, thisOutput+chunkOffset, nelem);
      }
      if (tid == 0)
        printf("%d: COMPLETED FOR peer %d\n", comm->rank, peer);
    };

    int edgeRank = butterfly->edgeRank, edge = 1 << log2i(comm->nRanks);
    bool edgeSend = (comm->rank & edge);
    if (edgeRank != -1)
      reduce(edgeRank, edgeSend, !edgeSend, 0);

    for (int p = 0; p < log2i(comm->nRanks); p++) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1)
        reduce(peer, true, true, p + (edgeRank != -1 ? 1 : 0));
    }

    if (edgeRank != -1)
      reduce(edgeRank, !edgeSend, edgeSend, log2i(comm->nRanks) + 1);
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
	using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runButterfly<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runButterfly<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runButterfly<T, RedOp, ProtoLL128>(args);
  }
};

#endif