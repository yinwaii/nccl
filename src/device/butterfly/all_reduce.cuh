#ifndef __BUTTERFLY_ALL_REDUCE_H__
#define __BUTTERFLY_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

__host__ __device__ static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runButterfly(ncclWorkElem *args) {
    printf("HERE is ALGORITHM for BUTEERFLY!\n");
    // const int tid = threadIdx.x;
    // const int nthreads = args->nThreads;
    // const int bid = args->coll.bid;
    // const int nChannels = args->coll.nChannels;
    // struct ncclDevComm* comm = args->comm;
    // struct ncclChannel* channel = comm->channels+blockIdx.x;
    // struct ncclButterfly* butterfly = &channel->butterfly;
    // ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    // const ssize_t minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    // const ssize_t loopSize = int(nChannels*chunkSize);
    // const ssize_t size = args->coll.count;

    // // Compute pointers
    // const T *__restrict__ thisInput = (const T *)args->sendbuff;
    // T *__restrict__ thisOutput = (T *)args->recvbuff;

    // for (int p = 0; p < log2i(comm->nRanks); p++) {
    //   int peer = butterfly->devPeerRanks[p];
    //   if (peer != -1) {
    //     Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
    //       prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm, 0);

    //       for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    //         ssize_t realChunkSize;
    //         if (Proto::Id == NCCL_PROTO_SIMPLE) {
    //           realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
    //           realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
    //         }
    //         else
    //           realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSize)*minChunkSize);
    //         realChunkSize = int(realChunkSize);

    //         ssize_t chunkOffset = gridOffset + bid * realChunkSize;
    //         int nelem = min(realChunkSize, size - chunkOffset);
    //         prims.send(thisInput+chunkOffset, nelem);
    //         prims.recvReduceCopy(thisInput+chunkOffset, thisOutput+chunkOffset, nelem);
    //       }
    //   }
    // }
    // if ((comm->nRanks & (-comm->nRanks)) != comm->nRanks) {
    //   for (int p = 0; p < comm->nRanks; p++){
    //     int peer = butterfly->devLastRanks[p];
    //     Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
    //         prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm, 0);
    //     for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    //       ssize_t realChunkSize;
    //       if (Proto::Id == NCCL_PROTO_SIMPLE) {
    //         realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
    //         realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
    //       }
    //       else
    //         realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSize)*minChunkSize);
    //       realChunkSize = int(realChunkSize);

    //       ssize_t chunkOffset = gridOffset + bid * realChunkSize;
    //       int nelem = min(realChunkSize, size - chunkOffset);
    //       if (comm->rank == 0)
    //         prims.send(thisInput+chunkOffset, nelem);
    //       else
    //         prims.recv(thisOutput+chunkOffset, nelem);
    //     }
    //   }
    // }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
	using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runButterfly<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runButterfly<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    runButterfly<T, RedOp, ProtoLL128>(args);
  }
};

#endif