

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
}
template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
		const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclButterfly* butterfly = &channel->butterfly;
		const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
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

    auto edgeReduce = [&]__device__(int peer, bool scatter, int step)->void {
      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 0, RedOp>
        prims(tid, nthreads, &peer, &peer, thisOutput, stepSize, channel, comm);
      if (tid == 0)
        printf("%d: START FOR peer %d\n", comm->rank, peer);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));

        ssize_t chunkOffset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, size - chunkOffset);
        if (tid == 0) {
          printf("%d: chunkOffset: %ld nelem: %d", comm->rank, chunkOffset, nelem);
        }
        if ((rank < peer) ^ !scatter)
          prims.send(thisInput+chunkOffset, nelem);
        else if (scatter)
          prims.recvReduceCopy(thisInput+chunkOffset, thisOutput+chunkOffset, nelem);
        else
          prims.recv(thisOutput+chunkOffset, nelem);
      }
      if (tid == 0)
        printf("%d: COMPLETED FOR peer %d\n", comm->rank, peer);
    };

    auto reduce = [&]__device__(int peer, bool scatter, int step)->void {
      int halfSize = size >> (step + 1);
			ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 0, RedOp>
        prims(tid, nthreads, &peer, &peer, thisOutput, stepSize, channel, comm);

      if (tid == 0)
        printf("%d: START FOR peer %d\n", comm->rank, peer);
      for (ssize_t gridOffset = commOffset; gridOffset < commOffset + halfSize; gridOffset += loopSize) {
				ssize_t realChunkSize = min(chunkSize, DIVUP(commOffset+halfSize-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));

        ssize_t chunkOffset = gridOffset + bid * realChunkSize;
        int nelem = min(realChunkSize, commOffset+halfSize - chunkOffset);
        if (tid == 0) {
          printf("%d: chunkOffset: %ld nelem: %d", comm->rank, chunkOffset, nelem);
        }

        prims.send(thisInput+chunkOffset+(((rank<peer)^scatter)?0:halfSize), nelem);
        if (scatter)
          prims.recvReduceCopy(thisInput+chunkOffset+(rank<peer?0:halfSize), thisOutput+chunkOffset+(rank<peer?0:halfSize), nelem);
        else
          prims.recv(thisOutput+chunkOffset+(rank<peer?halfSize:0), nelem);
      }
      prims.conRecv(thisOutput, 1);
      prims.conSend(thisInput, 1);
      if (tid == 0)
        printf("%d: COMPLETED FOR peer %d\n", comm->rank, peer);
    };

    int edgeRank = butterfly->edgeRank;
    if (edgeRank != -1)
      edgeReduce(edgeRank, true, log2i(comm->nRanks));

    for (int p = 0; p < log2i(comm->nRanks); p++) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        reduce(peer, true, p);
        if (rank > peer)
          commOffset += size >> (p + 1);
      }
    }
    for (int p = log2i(comm->nRanks) - 1; p >= 0; p--) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        if (rank > peer)
          commOffset -= size >> (p + 1);
        reduce(peer, false, p);
      }
    }

    if (edgeRank != -1)
      edgeReduce(edgeRank, false, log2i(comm->nRanks) + 1);
  }
};


#endif