#ifndef NCCL_PRIMS_SIMPLE_H_
#define NCCL_PRIMS_SIMPLE_H_
#include "primitives.cuh"
// Implementation of primitive types
template <typename T, typename RedOp, typename Fan, int Direct,
          int SlicePerChunk, int StepPerSlice, int Unroll>
class Primitives<T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>> {
 private:
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  const T* recvDirectBuff[MaxRecv];
  T* sendDirectBuff[MaxSend];
  const T* recvBuff[MaxRecv];
  T* sendBuff[MaxSend];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  inline __device__ void barrier() {
    if (MaxSend>MaxRecv) {
      asm volatile ("bar.sync 1, %0;" :: "r"(nthreads+WARP_SIZE));
    } else {
      asm volatile ("bar.sync 2, %0;" :: "r"(nthreads+WARP_SIZE));
    }
  }
  inline __device__ void subBarrier() {
    if (MaxSend>MaxRecv) {
      asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
    } else {
      asm volatile ("bar.sync 4, %0;" :: "r"(nthreads));
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + StepPerSlice) {
	//printf("wait send t:%d\n",tid);
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      //printf("wait send t:%d done\n",tid);
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += StepPerSlice;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + StepPerSlice) {
	//printf("wait recv t:%d\n",tid);
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      //printf("wait recv t:%d done\n",tid);
      recvConnTail += StepPerSlice;
    }
  }

  inline __device__ void confirmSend() {
    spins = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache < sendConnHead) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
    }
  }

  inline __device__ void confirmRecv() {
    spins = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += StepPerSlice;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += StepPerSlice;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += StepPerSlice;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += StepPerSlice;
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV>
  inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
    return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTSEND>
  inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
    return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset) {
    if (nelem == -1) {
      bool syncThread = tid >= nthreads;
      if (!syncThread) {
        confirmSend();
      }
      return;
    }
    if (nelem == -2) {
      bool syncThread = tid >= nthreads;
      if (!syncThread) {
        confirmRecv();
      }
      return;
    }
    int offset = 0;
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(nelem, 16*SlicePerChunk)*16, sliceSize/32);

    const T* srcs[RECV*MaxRecv+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<MaxRecv && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*MaxSend+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<MaxSend && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads;

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, MaxSend>(tid, nthreads, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<Unroll, RedOp, T, RECV+SRC, RECV*MaxRecv+SRC, SEND+DST, SEND*MaxSend+DST>(tid, nthreads, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*MaxRecv; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*MaxSend; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvBuff[i] = (const T*)conn->buffs[NCCL_PROTO_SIMPLE];
    recvStep[i] = conn->step;
    recvStep[i] = ROUNDUP(recvStep[i], SlicePerChunk*StepPerSlice);
    recvDirectBuff[i] = NULL;
    if (Direct && (conn->direct & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *conn->ptrExchange = directBuff;
    }
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
    sendBuff[i] = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    sendStep[i] = conn->step;
    sendStep[i] = ROUNDUP(sendStep[i], SlicePerChunk*StepPerSlice);
    sendDirectBuff[i] = NULL;
    if (Direct && (conn->direct & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = conn->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      barrier();
      if (tid == 0) *ptr = NULL;
    }
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
    }
    if (tid >= nthreads && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads && wid < nrecv) {
      recvConn->step = recvConnHead;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  Primitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, struct ncclChannel* channel, struct ncclDevComm* comm)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<MaxRecv && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<MaxSend && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ __forceinline__ void
  conSend(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, -1, 0);
  }

  __device__ __forceinline__ void
  conRecv(T *dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, -2, 0);
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ ~Primitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
#endif