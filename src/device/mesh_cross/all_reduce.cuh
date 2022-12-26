#ifndef __MESHCROSS_ALL_REDUCE_H__
#define __MESHCROSS_ALL_REDUCE_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ void runMeshCross(struct CollectiveArgs* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->coll.nThreads - Proto::Warp;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclMeshCross* meshCross = &channel->meshCross;
    ssize_t chunkSize = int(Proto::calcBytePerStep(comm)/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const ssize_t loopSize = nChannels*meshCross->nIntraRanks*meshCross->nInterRanks*(ssize_t)chunkSize;
    const ssize_t size = args->coll.count;

    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    // Compute pointers
    const T *__restrict__ thisInput = (const T *)args->sendbuff;
    T *__restrict__ thisOutput = (T *)args->recvbuff;

    auto getRealChunkSize = [&] __device__(ssize_t gridOffset, ssize_t tailOffset) -> ssize_t {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels*meshCross->nIntraRanks*meshCross->nInterRanks));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      }
      else
        realChunkSize = min(chunkSize, DIVUP(size-gridOffset, nChannels*meshCross->nIntraRanks*meshCross->nInterRanks*minChunkSize)*minChunkSize);
      return int(realChunkSize);
    };

    auto calcOffset = [&]__device__(ssize_t gridOffset, ssize_t realChunkSize, int chunk, int slice)->ssize_t {
      if (Proto::Id == NCCL_PROTO_SIMPLE)
        return gridOffset + bid*meshCross->nIntraRanks*meshCross->nInterRanks*realChunkSize + chunk*realChunkSize*meshCross->nInterRanks + slice * realChunkSize;
      else
        return gridOffset + (chunk*nChannels + bid)*realChunkSize*meshCross->nInterRanks + slice * realChunkSize;
    };

    auto interReduce = [&] __device__(Primitives<T, RedOp, FanSymmetric<1>, 1, Proto> &prims) -> void {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;
        int slice;
        int nSubRanks = meshCross->nIntraRanks / meshCross->nInterRanks;

        chunk = comm->rank / meshCross->nIntraRanks * nSubRanks + (comm->rank % nSubRanks);
        if (gridOffset == 0 && tid == 0)
          printf("%d --> %d\n", comm->rank, chunk);
        slice = meshCross->devInterRanks[meshCross->nInterRanks - 1] % meshCross->nInterRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        prims.send(thisOutput + offset, nelem);

        for (int j = 2; j < meshCross->nInterRanks; ++j) {
          slice = meshCross->devInterRanks[meshCross->nInterRanks - j] % meshCross->nInterRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          prims.recvReduceSend(thisOutput + offset, nelem);
        }

        slice = meshCross->devInterRanks[0] % meshCross->nInterRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        prims.directRecvReduceCopySend(thisOutput + offset, thisOutput + offset, offset, nelem);

        for (int j = 1; j < meshCross->nInterRanks - 1; ++j) {
          slice = meshCross->devInterRanks[meshCross->nInterRanks - j] % meshCross->nInterRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }

        slice = meshCross->devInterRanks[1] % meshCross->nInterRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        prims.directRecv(thisOutput + offset, offset, nelem);
      }
    };

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
      primIntra(tid, nthreads, &meshCross->intra_prev, &meshCross->intra_next, thisOutput, channel, comm);
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
      primMirror(tid, nthreads, &meshCross->mirror, &meshCross->mirror, thisOutput, channel, comm);

    if (tid == 0)
      printf("Begin...\n");

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      for (int slice = 0; slice < meshCross->nInterRanks; slice++) {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = meshCross->devIntraRanks[meshCross->nIntraRanks - 1] % meshCross->nIntraRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        primIntra.send(thisInput + offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j = 2; j < meshCross->nIntraRanks; ++j) {
          chunk = meshCross->devIntraRanks[meshCross->nIntraRanks - j] % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primIntra.recvReduceSend(thisInput + offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = meshCross->devIntraRanks[0] % meshCross->nIntraRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        primIntra.recvReduceCopy(thisInput + offset, thisOutput + offset, nelem);

        if (meshCross->mirror != comm->rank) {
          chunk = meshCross->devIntraRanks[0] % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primMirror.send(thisOutput + offset, nelem);

          chunk = meshCross->mirror % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primMirror.recv(thisOutput + offset, nelem);
        }
      }
    }

    if (tid == 0)
      printf("Step 3 begin...\n");

    if (meshCross->nInterRanks != meshCross->nIntraRanks) {
      Primitives<T, RedOp, FanSymmetric<1>, 1, Proto>
        primInter(tid, nthreads, &meshCross->inter_prev, &meshCross->inter_next, thisOutput, channel, comm);
      interReduce(primInter);
    }
    else
      interReduce(primIntra);

    if (tid == 0)
      printf("Step 3 end...\n");

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      for (int slice = 0; slice < meshCross->nInterRanks; slice++) {
        ssize_t realChunkSize = getRealChunkSize(gridOffset, size);
        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        if (meshCross->mirror != comm->rank) {
          chunk = meshCross->mirror % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primMirror.send(thisOutput + offset, nelem);

          chunk = meshCross->devIntraRanks[0] % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primMirror.recv(thisOutput + offset, nelem);
        }

        // step 0: push data to next GPU
        chunk = meshCross->devIntraRanks[0] % meshCross->nIntraRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        primIntra.directSend(thisOutput + offset, offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j = 1; j < meshCross->nIntraRanks - 1; ++j) {
          chunk = meshCross->devIntraRanks[meshCross->nIntraRanks - j] % meshCross->nIntraRanks;
          offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
          nelem = min(realChunkSize, size - offset);

          primIntra.directRecvCopySend(thisOutput + offset, offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = meshCross->devIntraRanks[1] % meshCross->nIntraRanks;
        offset = calcOffset(gridOffset, realChunkSize, chunk, slice);
        nelem = min(realChunkSize, size - offset);

        primIntra.directRecv(thisOutput + offset, offset, nelem);
      }
    }

    if (tid == 0)
      printf("End...\n");
  }
}

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_MESH_CROSS, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
	  using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, UNROLL>;
    runMeshCross<T, RedOp, Proto>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_MESH_CROSS, NCCL_PROTO_LL, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runMeshCross<T, RedOp, ProtoLL>(args);
  }
};

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclCollAllReduce, NCCL_ALGO_MESH_CROSS, NCCL_PROTO_LL128, RedOp, T, UNROLL> {
  public:
  __device__ void run(struct CollectiveArgs* args) {
    runMeshCross<T, RedOp, ProtoLL128>(args);
  }
};

#endif