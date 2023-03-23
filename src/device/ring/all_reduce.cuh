/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runRing(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const int chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = comm->nRanks;
    const ssize_t loopSize = nChannels*nranks*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

#if defined(ENABLE_NPKIT)
    int npKitCtxIdx = bid;
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      uint64_t* cpuTimestamp = comm->cpuTimestamp;
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_ENTRY, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, channel, comm);

#if defined(ENABLE_NPKIT)
    if (tid == 0) {
      prims.npKitCtxIdx = npKitCtxIdx;
    }
#endif

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels*nranks));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);

      auto calcOffset = [&]__device__(int chunk)->ssize_t {
        if (Proto::Id == NCCL_PROTO_SIMPLE)
          return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
          return gridOffset + (chunk*nChannels + bid)*realChunkSize;
      };

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      prims.send(thisInput+offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // k-2 steps: reduce and copy to next GPU

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSend(thisInput+offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = calcOffset(chunk);
      nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_EXIT, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};
