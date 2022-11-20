#ifndef __RING_SENDRECV_H__
#define __RING_SENDRECV_H__
#include "collectives.h"
#include "devcomm.h"
#include "primitives.cuh"

template<class RedOp, typename T, int UNROLL>
class ncclFunction<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, RedOp, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* firstArgs) {
      struct ncclWorkElem* args = firstArgs;
      int tid = threadIdx.x;
      int group = 0;
      for (int s=0; s<NCCL_MAX_WORK_ELEMENTS; s++) {
        int nThreadsSegment = args->p2p.nThreads;
        if (nThreadsSegment == 0) return; // Nothing else to do
        int groupRecv = group;
        group += 1;
        int groupSend = group;
        group += nThreadsSegment > 128 ? 2 : 1;
        if (tid < nThreadsSegment) {
          const int nThreads = nThreadsSegment > 128 ? nThreadsSegment-WARP_SIZE : nThreadsSegment;

          // Compute pointers
          const T* sendbuff = (const T*)args->sendbuff;
          T* recvbuff = (T*)args->recvbuff;
          const ssize_t sendCount = args->p2p.sendCount;
          const ssize_t recvCount = args->p2p.recvCount;

          const int delta = args->p2p.delta;
          if (delta == 0) {
            if (tid < nThreads && sendbuff != recvbuff) {
              // local copy : ReduceOrCopyMulti takes an int as number of elements,
              // so we split it in blocks of 1G elements.
              int blockSize = 1<<30;
              for (size_t offset=0; offset<sendCount; offset += blockSize) {
                size_t remaining = sendCount - offset;
                if (remaining < blockSize) blockSize = remaining;
                ReduceOrCopyMulti<UNROLL, RedOp, T, 1, 1, 1, 1>(tid, nThreads, RedOp(), false, false, 1, &sendbuff, 1, &recvbuff, blockSize);
                sendbuff += blockSize; recvbuff += blockSize;
              }
            }
          } else {
            struct ncclDevComm* comm = args->comm;
            struct ncclChannel* channel = comm->channels+blockIdx.x;

            const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/(sizeof(T)*NCCL_STEPS);
            const int chunkSize = stepSize/SENDRECV_SLICEFACTOR;

            int nThreadsSplit = nThreads/2;
            if ((tid < nThreadsSplit) && recvCount >= 0) {
              int peer = (comm->rank-delta+comm->nRanks)%comm->nRanks;
              int nt = nThreadsSplit;
              Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, ProtoSimple<1, 1, UNROLL>>
                prims(tid, nt, &peer, NULL, recvbuff, channel, comm, groupRecv);

              if (recvCount == 0) {
                prims.recv(recvbuff, 0);
              } else for (ssize_t offset = 0; offset < recvCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, recvCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, recvCount-offset);
                prims.directRecv(recvbuff+offset, offset, nelem);
              }
            }
            if ((tid >= nThreadsSplit) && sendCount >= 0) {
              int peer = (comm->rank+delta)%comm->nRanks;
              int nt = nThreads-nThreadsSplit;
              Primitives<T, RedOp, FanAsymmetric<0,1>, 1, ProtoSimple<1, 1, UNROLL>>
                prims(tid-nThreadsSplit, nt + (nt >= 64 ? WARP_SIZE : 0), NULL, &peer, recvbuff, channel, comm, groupSend);

              if (sendCount == 0) {
                prims.send(sendbuff, 0);
              } else for (ssize_t offset = 0; offset < sendCount; offset += chunkSize) {
                int realChunkSize = min(chunkSize, sendCount-offset);
                ALIGN_SIZE(realChunkSize, nt*sizeof(uint64_t)/sizeof(T));
                int nelem = min(realChunkSize, sendCount-offset);
                prims.directSend(sendbuff+offset, offset, nelem);
              }
            }
          }
        }
        tid -= nThreadsSegment;
        if (tid < 0) return;
        args++;
      }
    }
};

#endif