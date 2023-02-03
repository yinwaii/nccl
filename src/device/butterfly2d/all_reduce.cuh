#ifndef __BUTTERFLY_2D_ALL_REDUCE_H__
#define __BUTTERFLY_2D_ALL_REDUCE_H__
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
  __device__ void runButterfly2d(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclButterfly2D* butterfly = &channel->butterfly2d;
    ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const ssize_t minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    const ssize_t loopSize = int(nChannels*chunkSize)*butterfly->nIntraRanks;
    const ssize_t size = args->coll.count;
    int rank = comm->rank, commOffset = 0;

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;


    auto getRealChunkSize = [&] __device__(ssize_t gridOffset, ssize_t tailOffset) -> ssize_t {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(tailOffset-gridOffset,nChannels*butterfly->nIntraRanks));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, DIVUP(tailOffset-gridOffset, nChannels*butterfly->nIntraRanks*minChunkSize)*minChunkSize);
      return int(realChunkSize);
    };

		auto calcOffset = [&]__device__(ssize_t gridOffset, int realChunkSize, int chunk)->ssize_t {
			if (Proto::Id == NCCL_PROTO_SIMPLE)
				return gridOffset + bid*butterfly->nIntraRanks*realChunkSize + chunk*realChunkSize;
			else
				return gridOffset + (chunk*nChannels + bid)*realChunkSize;
		};

    // if (tid == 0)
    //   printf("Step 1: Intra-node reducescatter\n");
    if (butterfly->nIntraRanks > 1) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> primIntra(tid, nthreads, &butterfly->intra_prev, &butterfly->intra_next, thisOutput, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = butterfly->devIntraRanks[butterfly->nIntraRanks-1];
        offset = calcOffset(gridOffset, realChunkSize, chunk);
        nelem = min(realChunkSize, size-offset);

        primIntra.send(thisInput+offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<butterfly->nIntraRanks; ++j) {
          chunk = butterfly->devIntraRanks[butterfly->nIntraRanks-j];
          offset = calcOffset(gridOffset, realChunkSize, chunk);
          nelem = min(realChunkSize, size-offset);

          primIntra.recvReduceSend(thisInput+offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = butterfly->devIntraRanks[0];
        offset = calcOffset(gridOffset, realChunkSize, chunk);
        nelem = min(realChunkSize, size-offset);

        primIntra.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      }
    }

    // if (tid == 0)
    //   printf("Step 2: Inter-node allreduce\n");

    int edgeRank = butterfly->edgeRank;

    if (edgeRank != -1) {
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
        prims(tid, nthreads, &edgeRank, &edgeRank, thisOutput, channel, comm);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize)
      {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        ssize_t chunkOffset = calcOffset(gridOffset, realChunkSize, butterfly->devIntraRanks[0]);
        int nelem = min(realChunkSize, size - chunkOffset);

        if (rank < edgeRank)
          prims.send(thisInput+chunkOffset, nelem);
        else
          prims.recvReduceCopy(thisInput+chunkOffset, thisOutput+chunkOffset, nelem);
      }
    }

    for (int p = 0; p < butterfly->nPeerRanks; p++) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
          prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm);

        ssize_t length = size >> (p + 1);
        const T *stepInput = (p > 0 || edgeRank != -1 && rank > edgeRank) ? thisOutput : thisInput;
        
        for (ssize_t gridOffset = commOffset; gridOffset < commOffset + length; gridOffset += loopSize)
        {
          ssize_t realChunkSize = getRealChunkSize(gridOffset, commOffset + length);
          ssize_t chunkOffset = calcOffset(gridOffset, realChunkSize, butterfly->devIntraRanks[0]);
          int nelem = min(realChunkSize, commOffset + length - chunkOffset);
          if (rank < peer) {
            prims.send(stepInput + chunkOffset + length, nelem);
            prims.recvReduceCopy(stepInput + chunkOffset, thisOutput + chunkOffset, nelem);
          }
          else {
            prims.send(stepInput + chunkOffset, nelem);
            prims.recvReduceCopy(stepInput + chunkOffset + length, thisOutput + chunkOffset + length, nelem);
          }
        }

        if (rank > peer)
          commOffset += length;
        if (Proto::Id == NCCL_PROTO_SIMPLE) {
          prims.conRecv(thisOutput, 1);
          prims.conSend(thisInput, 1);
        }
      }
    }

    for (int p = butterfly->nPeerRanks - 1; p >= 0; p--) {
      int peer = butterfly->devPeerRanks[p];
      if (peer != -1) {
        Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
          prims(tid, nthreads, &peer, &peer, thisOutput, channel, comm);

        ssize_t length = size >> (p + 1);

        if (rank > peer)
          commOffset -= length;

        for (ssize_t gridOffset = commOffset; gridOffset < commOffset + length; gridOffset += loopSize)
        {
          ssize_t realChunkSize = getRealChunkSize(gridOffset, commOffset + length);
          ssize_t chunkOffset = calcOffset(gridOffset, realChunkSize, butterfly->devIntraRanks[0]);
          int nelem = min(realChunkSize, commOffset + length - chunkOffset);
          if (rank < peer) {
            prims.send(thisOutput + chunkOffset, nelem);
            prims.recv(thisOutput + chunkOffset + length, nelem);
          }
          else {
            prims.send(thisOutput + chunkOffset + length, nelem);
            prims.recv(thisOutput + chunkOffset, nelem);
          }
        }
        if (Proto::Id == NCCL_PROTO_SIMPLE) {
          prims.conRecv(thisOutput, 1);
          prims.conSend(thisInput, 1);
        }
      }
    }
    
    if (edgeRank != -1) {
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
        prims(tid, nthreads, &edgeRank, &edgeRank, thisOutput, channel, comm);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize)
      {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        ssize_t chunkOffset = calcOffset(gridOffset, realChunkSize, butterfly->devIntraRanks[0]);
        int nelem = min(realChunkSize, size - chunkOffset);

        if (rank < edgeRank)
          prims.recv(thisOutput+chunkOffset, nelem);
        else
          prims.send(thisOutput+chunkOffset, nelem);
      }
    }

    // if (tid == 0)
    //   printf("Step 3: Intra-node allgather\n");

    if (butterfly->nIntraRanks > 1) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> primIntra(tid, nthreads, &butterfly->intra_prev, &butterfly->intra_next, thisOutput, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = butterfly->devIntraRanks[0];
        offset = calcOffset(gridOffset, realChunkSize, chunk);
        nelem = min(realChunkSize, size-offset);

        primIntra.directSend(thisOutput+offset, offset, nelem);

        // k-2 steps: copy to next GPU
        for (int j=1; j<butterfly->nIntraRanks-1; ++j) {
          chunk = butterfly->devIntraRanks[butterfly->nIntraRanks - j];
          offset = calcOffset(gridOffset, realChunkSize, chunk);
          nelem = min(realChunkSize, size-offset);

          primIntra.directRecvCopySend(thisOutput+offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        chunk = butterfly->devIntraRanks[1];
        offset = calcOffset(gridOffset, realChunkSize, chunk);
        nelem = min(realChunkSize, size-offset);

        // Final wait/copy.
        primIntra.directRecv(thisOutput+offset, offset, nelem);
      }
    }
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY_2D, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
	using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runButterfly2d<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY_2D, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runButterfly2d<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_BUTTERFLY_2D, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runButterfly2d<T, RedOp, ProtoLL128>(args);
  }
};

#endif