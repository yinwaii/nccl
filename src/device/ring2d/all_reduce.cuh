#ifndef __RING_2D_ALL_REDUCE_H__
#define __RING_2D_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit.h"
#endif

namespace {
//   __host__ __device__ static long log2i(long n) {
//     long l = 0;
//     while (n >>= 1) l++;
//     return l;
//   }
  template<typename T, typename RedOp, typename Proto>
  __device__ void runRing2d(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing2D* ring = &channel->ring2d;
    ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const ssize_t minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    const ssize_t loopSize = int(nChannels*chunkSize)*ring->nIntraRanks*ring->nInterRanks;
    const ssize_t size = args->coll.count;
    int rank = comm->rank, commOffset = 0;

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

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;


    auto getRealChunkSize = [&] __device__(ssize_t gridOffset, ssize_t tailOffset) -> ssize_t {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(tailOffset-gridOffset,nChannels*ring->nIntraRanks*ring->nInterRanks));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, DIVUP(tailOffset-gridOffset, nChannels*ring->nIntraRanks*ring->nInterRanks*minChunkSize)*minChunkSize);
      return int(realChunkSize);
    };

		auto calcOffset = [&]__device__(ssize_t gridOffset, int realChunkSize, int chunk, int slice)->ssize_t {
			if (Proto::Id == NCCL_PROTO_SIMPLE)
				return gridOffset + bid*ring->nIntraRanks*ring->nInterRanks*realChunkSize + chunk*ring->nInterRanks*realChunkSize + slice*realChunkSize;
			else
				return gridOffset + (chunk*nChannels + bid)*realChunkSize;
		};

    if (tid == 0)
      printf("Step 1: Intra-node reducescatter\n");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_REDUCESCATTER_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_REDUCESCATTER_ENTRY, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (ring->nIntraRanks > 1) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> primIntra(tid, nthreads, &ring->intra_prev, &ring->intra_next, thisOutput, channel, comm);
#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        primIntra.npKitCtxIdx = npKitCtxIdx;
      }
#endif
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
				for (int slice = 0; slice < ring->nInterRanks; slice++) {
					ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
					/////////////// begin AllReduce steps ///////////////
					ssize_t offset;
					int nelem;
					int chunk;

					// step 0: push data to next GPU
					chunk = ring->devIntraRanks[ring->nIntraRanks-1];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

					primIntra.send(thisInput+offset, nelem);

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

					for (int j=2; j<ring->nIntraRanks; ++j) {
						chunk = ring->devIntraRanks[ring->nIntraRanks-j];
						offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
						nelem = min(realChunkSize, size-offset);

						primIntra.recvReduceSend(thisInput+offset, nelem);
					}

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

					// step k-1: reduce this buffer and data, which will produce the final
					// result that we store in this data and push to the next GPU
					chunk = ring->devIntraRanks[0];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_COPY_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_COPY_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

					primIntra.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_COPY_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_COPY_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif
				}
      }
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_REDUCESCATTER_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_REDUCESCATTER_EXIT, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (tid == 0)
      printf("Step 2: Inter-node allreduce\n");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTER_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTER_ENTRY, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
		
		if (ring->nInterRanks > 1) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> primInter(tid, nthreads, &ring->inter_prev, &ring->inter_next, thisOutput, channel, comm);
#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        primInter.npKitCtxIdx = npKitCtxIdx;
      }
#endif
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
				ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
				/////////////// begin AllReduce steps ///////////////
				ssize_t offset;
				int nelem;
				int chunk = ring->devIntraRanks[0];
				int slice;

				// step 0: push data to next GPU
				slice = ring->devInterRanks[ring->nInterRanks-1];
				offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
				nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

				primInter.send(thisInput+offset, nelem);

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

				for (int j=2; j<ring->nInterRanks; ++j) {
					slice = ring->devInterRanks[ring->nInterRanks-j];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

					primInter.recvReduceSend(thisInput+offset, nelem);
				}

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_RECV_REDUCE_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

				// step k-1: reduce this buffer and data, which will produce the final
				// result that we store in this data and push to the next GPU
				
				slice = ring->devInterRanks[0];
				offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
				nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_REDUCE_COPY_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

				primInter.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

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
				for (int j=1; j<ring->nInterRanks-1; ++j) {
					chunk = ring->devInterRanks[ring->nInterRanks-j];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

					primInter.directRecvCopySend(thisOutput+offset, offset, nelem);
				}

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

				// Make final copy from buffer to dest.
				chunk = ring->devInterRanks[1];
				offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
				nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

				// Final wait/copy.
				primInter.directRecv(thisOutput+offset, offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

      }
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTER_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTER_EXIT, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (tid == 0)
      printf("Step 3: Intra-node allgather\n");

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_ALLGATHER_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_ALLGATHER_ENTRY, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif

    if (ring->nIntraRanks > 1) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> primIntra(tid, nthreads, &ring->intra_prev, &ring->intra_next, thisOutput, channel, comm);
#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        primIntra.npKitCtxIdx = npKitCtxIdx;
      }
#endif
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
				for (int slice = 0; slice < ring->nInterRanks; slice++) {
					ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
					/////////////// begin AllReduce steps ///////////////
					ssize_t offset;
					int nelem;
					int chunk;

					// step k-1: reduce this buffer and data, which will produce the final
					// result that we store in this data and push to the next GPU
					chunk = ring->devIntraRanks[0];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_SEND_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_SEND_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

					primIntra.directSend(thisOutput+offset, offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_SEND_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
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
					for (int j=1; j<ring->nIntraRanks-1; ++j) {
						chunk = ring->devIntraRanks[ring->nIntraRanks - j];
						offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
						nelem = min(realChunkSize, size-offset);

						primIntra.directRecvCopySend(thisOutput+offset, offset, nelem);
					}

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT)
      if (tid == 0 && nranks > 2) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

					// Make final copy from buffer to dest.
					chunk = ring->devIntraRanks[1];
					offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
					nelem = min(realChunkSize, size-offset);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
        prims.npKitDataProcessTotalTime = 0;
      }
#endif

					// Final wait/copy.
					primIntra.directRecv(thisOutput+offset, offset, nelem);

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, clock64(),
            comm->npKitEventCollectContexts + npKitCtxIdx);
      }
#endif

				}
      }
    }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_ALLGATHER_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_REDUCE_RING2D_INTRA_ALLGATHER_EXIT, size*sizeof(T), 0, clock64(),
          comm->npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING_2D, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
	using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runRing2d<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING_2D, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runRing2d<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_RING_2D, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runRing2d<T, RedOp, ProtoLL128>(args);
  }
};

#endif