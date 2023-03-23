/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "align.h"
#if defined(ENABLE_NPKIT)
#include "npkit/npkit_struct.h"
#endif
#include "algo_config.h"
#include <stdint.h>

#define NCCL_NUM_FUNCTIONS 5 // SendRecv not included for now
typedef enum { ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollSendRecv} ncclFunc_t;
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define WARP_SIZE 32
#define NCCL_MAX_NTHREADS 512
#define NCCL_LL_MAX_NTHREADS NCCL_MAX_NTHREADS
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX   0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_DIRECT_GPU 0x01
#define NCCL_DIRECT_NIC 0x10

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
  struct ncclComm *comm;
};

struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclDevComm;

/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclDevComm* comm;

  // local and remote input, output, and buffer
  const void * sendbuff;
  void * recvbuff;

  // Op-specific fields. Make sure the common part stays the
  // same on all structs of the union
  union {
    struct {
      uint16_t nThreads;
    } common;
    struct {
      uint16_t nThreads;
      uint8_t bid;
      uint8_t nChannels;
      uint32_t root;
      size_t count;
      size_t lastChunkSize;
    } coll;
    struct {
      uint16_t nThreads;
      uint16_t unused;
      int32_t delta;
      size_t sendCount;
      size_t recvCount;
    } p2p;
  };
};
struct ncclColl {
  union {
    struct {
      struct CollectiveArgs args;
      uint16_t funcIndex;
      uint16_t nextIndex;
      uint8_t  active;
    };
    int data[0x10];
  };
};
static_assert(sizeof(struct ncclColl) == (0x10*sizeof(int)), "ncclColl must have a pow2 size");

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct ncclChannel* channels;

#if defined(ENABLE_NPKIT)
  NpKitEventCollectContext* npKitEventCollectContexts;
  uint64_t* cpuTimestamp;
#endif
};

#endif
