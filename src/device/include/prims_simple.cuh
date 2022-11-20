#ifndef __PRIMS_SIMPLE_H__
#define __PRIMS_SIMPLE_H__

// Implementation of primitive types
template <typename T, typename RedOp, typename Fan, int Direct,
          int SlicePerChunk, int StepPerSlice, int Unroll>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>>
{
private:
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input = 0, Output = 1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       PtrsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       DirectEnabled = 0x200,
                       ThreadsSynced = 0x400;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  struct ncclConnInfo* conn = NULL;
  volatile int* connSizesFifoPtr = NULL;
  void** connPtrsFifoPtr = NULL;
  volatile uint64_t* connHeadPtr = NULL;
  volatile uint64_t* connTailPtr = NULL;
  uint64_t connTailCache; // Cache last seen value
  uint64_t connHeadCache; // Cache last seen value

   int index; // Peer index I'm responsible for
   int flags = 0;
   int peer = -1;
   int group;
   uint64_t step;
   T *direct = NULL;
   T *buff;
   struct ncclDevComm *comm;

   const T **srcs;
   T **dsts;

   // Don't use barrier 0 as it's used by the final sync
   inline __device__ void barrier() {
     if (nthreads == WARP_SIZE)
       __syncwarp();
     else
      asm volatile("bar.sync %0, %1;" :: "r"(group+1), "r"(nthreads));
     // flags |= ThreadsSynced;
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads)
      barrier();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(group+2), "r"(nworkers));
  }

  uint32_t spins = 0;

  inline __device__ bool checkAbort() {
    spins++;
    if (!(flags & Aborted) && spins == SPINS_BEFORE_CHECK_ABORT) {
      flags |= *(comm->abortFlag) ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  inline __device__ void waitPeer(intptr_t dstIx, intptr_t remoteOutIx, int offset, int nelts) {
    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      bool const isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
      spins = 0;
      uint64_t *connStepCache = isSendNotRecv ? &connHeadCache : &connTailCache;
      volatile uint64_t *connStepPtr = isSendNotRecv ? connHeadPtr : connTailPtr;
      while (*connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        *connStepCache = *connStepPtr;
        if (checkAbort()) break;
      }
      if (isSendNotRecv && connSizesFifoPtr)
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);

      const T **ptrs = isSendNotRecv ? (dsts + Dst)
                                  : (srcs + Src);
      if (connPtrsFifoPtr)
        loadPtr(connPtrsFifoPtr + step%NCCL_STEPS, ptrs[index]);
      else if ((isSendNotRecv ? DirectSend : DirectRecv) && direct)
        ptrs[index] = direct + (isSendNotRecv ? remoteOutIx : dstIx) + offset;
      else
        ptrs[index] = buff+(step%NCCL_STEPS)*stepSize;
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv))
      *connHeadPtr = step += StepPerSlice;
    if (flags & (Send*RolePostSend))
      *connTailPtr = step += StepPerSlice;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  inline __device__ void GenericOp(
      const T* srcPtr, T* dstPtr, int nelem, ssize_t directOffset
    ) {
    // constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    // constexpr int DirectSend = 1 && Direct && DirectSend1;
    // constexpr int Src = SrcBuf != -1;
    // constexpr int Dst = DstBuf != -1;
    int sliceSize = stepSize * StepPerSlice;
    sliceSize = max(DIVUP(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int offset = 0;

      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
#pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      sliceSize = max(0, min(sliceSize, nelem-offset));
      if (tid < nworkers) {
        if (Src && (flags & RoleInput)) srcs[0] = srcPtr+offset;
        if (Dst && (flags & RoleOutput)) dsts[0] = dstPtr+offset;
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(directOffset, directOffset, offset, sliceSize);
        if (sliceSize > 0) {
          subBarrier();
          if (DirectRecv && srcs[0] == dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (Send) {
              // (1-Send) is only there to avoid compilation errors in case MaxSend=0 (and SEND=0).
              ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend>
                (tid, nworkers, RedOp(), false, false, 
                 1, srcs,
                fan.nsend(), dsts+1,
                sliceSize);
            }
          } else {
            ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst>
              (tid, nworkers, RedOp(), false, false, 
               Recv*fan.nrecv()+Src, srcs, 
               Send*fan.nsend()+Dst, dsts,
               sliceSize);
          }
        }
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      if (Send && (flags & RolePostSend) && sliceSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += sliceSize;
    }
  }


  __device__ __forceinline__ void loadRecvConn(struct ncclChannel* channel, T* directBuff) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      conn = &channel->devPeers[peer].recv.conn;
      step = conn->step;
      step = ROUNDUP(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostRecv) {
        connHeadPtr = conn->head;
        *connHeadPtr = step; // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (Direct && (conn->direct & NCCL_DIRECT_GPU)) {
          direct = directBuff;
          *conn->ptrExchange = directBuff;
        }
        connTailPtr = conn->tail;
        connTailCache = *connTailPtr;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclChannel* channel) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      conn = &channel->devPeers[peer].send.conn;
      step = conn->step;
      step = ROUNDUP(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostSend) {
        connTailPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        buff = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
        if (Direct && (conn->direct & NCCL_DIRECT_GPU)) {
          void* volatile* ptr = conn->ptrExchange;
          while ((direct = (T*)(*ptr)) == NULL);
          *ptr = NULL;
        }
        connHeadPtr = conn->head;
        connHeadCache = *connHeadPtr;
        connSizesFifoPtr = conn->sizesFifo;
        connPtrsFifoPtr = conn->ptrsFifo;
      }
    }
  }

 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers, 
      T* directBuff, struct ncclChannel* channel, struct ncclDevComm* comm, int group = 0, bool p2p = false
      ): 
    comm(comm), 
    tid(tid), 
    stepSize(comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)), 
    srcs((const T**)ncclShmem->ptrs[group].srcs), 
    dsts((T**)ncclShmem->ptrs[group].dsts),
    group(group) {

    // if p2p is true, then we use 2nd argument as nworkers, otherwise as nthreads
    if (p2p) {
      nworkers = nthreads;
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      int postThreads = MaxSend && nworkers >= 64 ? WARP_SIZE : 0;
      this->nthreads = nworkers + postThreads;
    }
    else {
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      this->nthreads = nthreads;
      this->nworkers = nthreads - (MaxSend && nworkers >= 64 ? WARP_SIZE : 0);
    }

    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend);

    constexpr int ThreadPerSync = 8;
    static_assert(MaxSend < ThreadPerSync && MaxRecv < ThreadPerSync, "Not enough threads to cover all peers");

    int g = tid / ThreadPerSync;
    int ng = nthreads / ThreadPerSync;
    index = tid % ThreadPerSync;

    if (g == 0) {
      if (index < nrecv) flags |= RoleWaitRecv;
      if (index == nrecv) flags |= RoleInput;
    } else if (g == 1) {
      if (index < nsend) flags |= RoleWaitSend;
      if (index == nsend) flags |= RoleOutput;
    } else if (g == ng - 2) {
      if (index < nrecv) flags |= RolePostRecv;
    } else if (g == ng - 1) {
      if (index < nsend) flags |= RolePostSend;
    }

    if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
    if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

    loadRecvConn(channel, directBuff);
    loadSendConn(channel);
  }

    __device__ __forceinline__ ~Primitives() {
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      conn->step = step;
      __threadfence_system();
    }
  }



  __device__ __forceinline__ void send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, nelem, 0);
  }
  __device__ __forceinline__ void directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0>(src, NULL, nelem, directOffset);
  }

  __device__ __forceinline__ void recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1>(src, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, nelem, 0);
  }
  __device__ __forceinline__ void directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1>(NULL, dst, nelem, directOffset);
  }

  __device__ __forceinline__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, nelem, 0);
  }

  __device__ __forceinline__ void recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, nelem, 0);
  }

  __device__ __forceinline__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, nelem, 0);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1>(src, dst, nelem, directOffset);
  }
};

#endif