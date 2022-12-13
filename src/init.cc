/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "algo_interface.h"
#include "interface.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define STR2(v) #v
#define STR(v) STR2(v)

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);

ncclNet_t* ncclNet = NULL;
ncclCollNet_t* ncclCollNet = NULL;

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initCollNet(ncclCollNet_t* collnet) {
  int ndev;
  if (collnet->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (collnet->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initNetPlugin(ncclNet_t** net, ncclCollNet_t** collnet) {
  void* netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == NULL) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : No plugin found (libnccl-net.so), using internal implementation");
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }
  ncclNet_t* extNet = (ncclNet_t*) dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
  if (extNet == NULL) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find " STR(NCCL_PLUGIN_SYMBOL) " symbol.");
  } else if (initNet(extNet) == ncclSuccess) {
    *net = extNet;
    // Check for CollNet
    ncclCollNet_t* extCollNet = (ncclCollNet_t*) dlsym(netPluginLib, STR(NCCL_COLLNET_PLUGIN_SYMBOL));
    if (extCollNet == NULL) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find " STR(NCCL_COLLNET_PLUGIN_SYMBOL) " symbol.");
    } else if (initCollNet(extCollNet) == ncclSuccess) {
      *collnet = extCollNet;
    }
    return ncclSuccess;
  }
  if (netPluginLib != NULL) dlclose(netPluginLib);
  return ncclSuccess;
}

ncclResult_t initNet() {
  // Always initialize bootstrap network
  NCCLCHECK(bootstrapNetInit());

  NCCLCHECK(initNetPlugin(&ncclNet, &ncclCollNet));
  if (ncclNet != NULL) return ncclSuccess;
  if (initNet(&ncclNetIb) == ncclSuccess) {
    ncclNet = &ncclNetIb;
  } else {
    NCCLCHECK(initNet(&ncclNetSocket));
    ncclNet = &ncclNetSocket;
  }
  return ncclSuccess;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    NCCLCHECK(initNet());
    INFO(NCCL_INIT, "Using network %s", ncclNetName());
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
}

#undef NCCL_NO_OPTIMIZE

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;
  free(comm->p2plist.peerlist);
  free(comm->p2plist.connect.recv);
  free(comm->p2plist.connect.send);

  free(comm->peerInfo);
  ncclTopoFree(comm->topo);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  CUDACHECK(cudaFree(comm->hostDevComm.channels));
  CUDACHECK(cudaFree(comm->devComm));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->doneEvent != NULL)
    CUDACHECK(cudaEventDestroy(comm->doneEvent));

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamDestroy(comm->groupStream));
  }

  // Last rank frees shared resources between threads
  int isLast;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
  if (isLast) {
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraCudaDevs);
    free(comm->intraCGMode);
    free(comm->intraCC);
  }
  CUDACHECK(cudaFreeHost((void *)comm->abortFlag));

  // Poison comm to try and catch a double free
  commPoison(comm);

  free(comm);
  return ncclSuccess;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  cudaEvent_t doneEvent;
  CUDACHECK(cudaEventCreateWithFlags(&doneEvent, cudaEventDisableTiming));

  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  comm->rank = comm->hostDevComm.rank =rank;
  comm->nRanks = comm->hostDevComm.nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %x", comm, rank, ndev, comm->cudaDev, comm->busId);

  comm->doneEvent = doneEvent;
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
#if CUDART_VERSION >= 9020
  comm->groupCudaStream = ncclParamGroupCudaStream();
#else
  // Don't allow the user to overload the default setting in older CUDA builds
  comm->groupCudaStream = NCCL_GROUP_CUDA_STREAM;
#endif
  comm->fatalError = ncclSuccess;

  NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
  comm->hostDevComm.abortFlag = comm->abortFlag;
  *comm->abortFlag = 0;

  comm->argsptr = &comm->args;
  comm->collNetSupport = 0;
  comm->p2plist.count=0;
  NCCLCHECK(ncclCalloc(&comm->p2plist.peerlist, comm->nRanks));
  for (int r=0; r<comm->nRanks; r++) comm->p2plist.peerlist[r].sendbytes = comm->p2plist.peerlist[r].recvbytes = -1;
  NCCLCHECK(ncclCalloc(&comm->p2plist.connect.recv, MAXCHANNELS*comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->p2plist.connect.send, MAXCHANNELS*comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c<MAXCHANNELS; c++) comm->channels[c].id = -1;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Duplicate the channels on the device
  NCCLCHECK(ncclCudaCalloc(&comm->hostDevComm.channels, comm->p2pnChannels));
  NCCLCHECK(ncclCudaMemcpy(comm->hostDevComm.channels, comm->channels, comm->p2pnChannels));

  // Copy userRanks and peers
  for (int r=0; r<comm->p2pnChannels; r++) {
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].ring.devUserRanks, comm->channels[r].ring.userRanks, comm->nRanks));
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].butterfly.devPeerRanks, comm->channels[r].butterfly.peerRanks, log2i(comm->nRanks)));
  }

  // Duplicate the dev comm on the device
  NCCLCHECK(ncclCudaCalloc(&comm->devComm, 1));
  NCCLCHECK(ncclCudaMemcpy(comm->devComm, &comm->hostDevComm, 1));
  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= NCCL_LOG_VERSION) {
    printf("%s\n", VERSION_STRING);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s", VERSION_STRING); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(&info->gdrSupport));
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  return ncclSuccess;
}

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

ncclResult_t initParams(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams = comm->intraParams+comm->intraRank;
  params->args = &comm->argsptr;
  params->stream = NULL;
  params->sharedMem = 0;
  params->blockDim.x = 0; params->blockDim.y = params->blockDim.z = 1;
  params->gridDim.x = 0; params->gridDim.y = params->gridDim.z = 1;
  return ncclSuccess;
}

// Allocate/Set Intra Process Structures and set CG options
ncclResult_t ncclCommSetIntra(struct ncclComm* comm, int rank, int ranks, struct ncclComm* comm0) {
  comm->intraRank = rank;
  comm->intraRanks = ranks;
  comm->intraPhase = 0;

  // Alloc shared structures
  if (rank == 0) {
    assert(comm == comm0);
    int* bar;
    NCCLCHECK(ncclCalloc(&bar, 2));
    bar[0] = bar[1] = 0;
    comm->intraBarrier = bar;
    NCCLCHECK(ncclCalloc(&comm->intraParams, comm->intraRanks));
    NCCLCHECK(ncclCalloc(&comm->intraCudaDevs, comm->intraRanks));
    int* CGMode;
    NCCLCHECK(ncclCalloc(&CGMode, 1));
    *CGMode = 0x11;
    comm->intraCGMode = CGMode;
    int* CC;
    NCCLCHECK(ncclCalloc(&CC, 1));
    *CC = ncclCudaCompCap();
    comm->intraCC = CC;
  } else {
    comm->intraBarrier = (int*)waitForNonNullPtr(&comm0->intraBarrier);
    comm->intraParams = (struct cudaLaunchParams*)waitForNonNullPtr(&comm0->intraParams);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
    comm->intraCC = (int*)waitForNonNullPtr(&comm0->intraCC);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;
  NCCLCHECK(initParams(comm));

  int cgMdLaunch = 0;

  // Set CG Mode
  comm->launchMode = ncclComm::GROUP;
  char* str = getenv("NCCL_LAUNCH_MODE");
  if (str) INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", str);
  if (comm->intraRanks == 1 || (str && strcmp(str, "PARALLEL") == 0)) {
    comm->launchMode = ncclComm::PARALLEL;
  }
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamCreateWithFlags(&comm->groupStream, cudaStreamNonBlocking));
#if CUDART_VERSION >= 9000
    if (*comm->intraCC && (ncclCudaCompCap() == *comm->intraCC)) {
      // Check whether the GPU supports Cooperative Group Multi Device Launch
      (void) cudaDeviceGetAttribute(&cgMdLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, comm->cudaDev);
    }
#endif
  }

  // Disable cgMdLaunch if any rank does not support it
  if (cgMdLaunch == 0) {
    *comm->intraCGMode = 0x10;
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1LL << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1LL << 20) /* 1MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = comm->hostDevComm.buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }
  return ncclSuccess;
}

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 3 AllGathers
  // 1. { peerInfo, comm }
  // 2. ConnectTransport[nranks], ConnectValue[nranks]
  // 3. { nThreads, nrings, compCap, prev[MAXCHANNELS], next[MAXCHANNELS] }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap));

  // AllGather1 - begin
  struct {
    struct ncclPeerInfo peerInfo;
    struct ncclComm* comm;
  } *allGather1Data;

  NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  allGather1Data[rank].comm = comm;
  struct ncclPeerInfo* myInfo = &allGather1Data[rank].peerInfo;
  NCCLCHECK(fillInfo(comm, myInfo, commHash));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather1Data, sizeof(*allGather1Data)));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks+1)); // Extra rank to represent CollNet root
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
    if ((i != rank) && (comm->peerInfo[i].hostHash == myInfo->hostHash) && (comm->peerInfo[i].busId == myInfo->busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %x", rank, i, myInfo->busId);
      return ncclInvalidUsage;
    }
  }
  // AllGather1 data is used again below
  // AllGather1 - end

  // Topo detection / System graph creation
  NCCLCHECK(ncclTopoGetSystem(comm, &comm->topo));
  // Compute paths between GPUs and NICs
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECK(ncclTopoTrimSystem(comm->topo, comm));
  // Recompute paths after trimming
  NCCLCHECK(ncclTopoComputePaths(comm->topo, comm->peerInfo));
  // Init search
  NCCLCHECK(ncclTopoSearchInit(comm->topo));
  // Print final topology
  NCCLCHECK(ncclTopoPrint(comm->topo));

  AlgoInfo<ncclTopoAlgo> algos = ncclTopoAlgos(comm);
  NCCLCHECK(ncclTopoInit(algos));

  struct ncclTopoGraph *graphs[NCCL_NUM_ALGORITHMS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    graphs[a] = &(algos[a]->graph);

  if (comm->rank == ncclParamGraphDumpFileRank())
    NCCLCHECK(ncclTopoDumpGraphs(comm->topo, NCCL_NUM_ALGORITHMS, graphs));

  // AllGather3 - begin
  struct {
    int cudaCompCap;
    int fullCudaCompCap;
    int nChannels;
    struct ncclGraphInfo graphInfos[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
  allGather3Data[rank].cudaCompCap = ncclCudaCompCap();
  allGather3Data[rank].nChannels = comm->nChannels = algos[NCCL_ALGO_RING]->graph.nChannels;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    algos[a]->graph.nChannels = comm->nChannels;
    NCCLCHECK(algos[a]->graphCopy(allGather3Data[rank].graphInfos + a));
  }

  NCCLCHECK(ncclTopoPreset(comm, algos, &allGather3Data[rank].topoRanks));

  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Determine nNodes, firstRanks, ...
  int* nodesFirstRank;
  NCCLCHECK(ncclCalloc(&nodesFirstRank, nranks));
  for (int i=0; i<nranks; i++) {
    int node = -1;
    int firstRank = allGather3Data[i].topoRanks.ringRecv[0];
    for (int n=0; n<comm->nNodes; n++) {
      if (nodesFirstRank[n] == firstRank) node = n;
    }
    if (node == -1) {
      node = comm->nNodes++;
      nodesFirstRank[node] = firstRank;
    }
    if (i == comm->rank) comm->node = node;
  }

  // Determine the minimum CUDA Compute capability of all GPUs
  int myCompCap = allGather3Data[rank].cudaCompCap;
  int minCompCap = myCompCap, maxCompCap = myCompCap;
  for (int i = 0; i < nranks; i++) {
    minCompCap = std::min(allGather3Data[i].cudaCompCap, minCompCap);
    maxCompCap = std::max(allGather3Data[i].cudaCompCap, maxCompCap);
  }

  int nChannelsOrig = comm->nChannels;
  struct ncclTopoRanks** allTopoRanks;
  NCCLCHECK(ncclCalloc(&allTopoRanks, comm->nRanks));
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      algos[a]->graph.nChannels = comm->nChannels = std::min(allGather3Data[i].nChannels, comm->nChannels);
      NCCLCHECK(algos[a]->graphFit(allGather3Data[i].graphInfos + a));
    }
  }

  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  NCCLCHECK(ncclTopoPostset(comm, algos, nodesFirstRank, allTopoRanks));

  free(allTopoRanks);
  free(nodesFirstRank);
  free(allGather3Data);

  // AllGather3 - end

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  NCCLCHECK(ncclTopoTuneModel(comm, minCompCap, maxCompCap, algos));

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* treeUp = &comm->channels[c].treeUp;
    struct ncclTree* treeDn = &comm->channels[c].treeDn;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d|%d->%d->%d/%d/%d",
        c, treeUp->down[0], treeUp->down[1], treeUp->down[2], rank, treeUp->up,
        treeDn->up, rank, treeDn->down[0], treeDn->down[1], treeDn->down[2]);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
  NCCLCHECK(ncclTopoSetAffinity(comm->topo, comm->rank));
  ncclResult_t ret;

  NCCLCHECKGOTO(computeBuffSizes(comm), ret, affinity_restore);

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
    NCCLCHECKGOTO(initChannel(comm, c), ret, affinity_restore);
  }

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    NCCLCHECKGOTO(algos[a]->transportSetup(), ret, affinity_restore);
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, affinity_restore);
  
  do {
    // Compute intra ranks (using AllGather1 data)
    int intraRank0 = -1, intraRank = -1, intraRanks = 0;
    for (int i = 0; i < nranks; i++) {
      if ((allGather1Data[i].peerInfo.hostHash == allGather1Data[rank].peerInfo.hostHash) &&
          (allGather1Data[i].peerInfo.pidHash == allGather1Data[rank].peerInfo.pidHash)) {
        if (intraRanks == 0) intraRank0 = i;
        if (i == rank) intraRank = intraRanks;
        intraRanks++;
      }
    }
    TRACE(NCCL_INIT,"hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
          rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
    if (intraRank == -1 || intraRank0 == -1 || allGather1Data[intraRank0].comm == NULL) {
      WARN("Failed to determine intra ranks hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
          rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
      return ncclInternalError;
    }
    NCCLCHECK(ncclCommSetIntra(comm, intraRank, intraRanks, allGather1Data[intraRank0].comm));
  } while(0);

  // Done with AllGather1 data
  free(allGather1Data);

  if (comm->nNodes) NCCLCHECK(ncclProxyCreate(comm));

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
affinity_restore:
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  if (ret != ncclSuccess) return ret;

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;

  CUDACHECK(cudaSetDevice(cudaDev));
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %x - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId);

  return ncclSuccess;
cleanup:
  if ((*newcomm) && (*newcomm)->bootstrap) bootstrapAbort((*newcomm)->bootstrap);
  *newcomm = NULL;
  return res;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev) {
  ncclResult_t res;
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    NCCLCHECKGOTO(bootstrapCreateRoot(&commId, true), res, end);
  }

  NCCLCHECKGOTO(ncclInit(), res, end);
  if (myrank == 0) showVersion();

  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, end);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, end);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto end;
  }

  if (ncclAsyncMode()) {
    NCCLCHECKGOTO(ncclAsyncInit(ncclCommInitRankSync, newcomm, nranks, commId, myrank, cudaDev), res, end);
  } else {
    NCCLCHECKGOTO(ncclCommInitRankSync(newcomm, nranks, commId, myrank, cudaDev), res, end);
  }
end:
  if (ncclAsyncMode()) return ncclAsyncErrCheck(res);
  else return res;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  ncclUniqueId uniqueId;
  NCCLCHECK(ncclGetUniqueId(&uniqueId));
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i);
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

static ncclResult_t commDestroy(ncclComm_t comm) {
  int savedDevice;
#ifdef ENABLE_TRACE
  int rank = comm->rank;
#endif
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d fatalError %d", comm, rank, *comm->abortFlag, comm->fatalError);

  CUDACHECK(cudaStreamSynchronize(comm->groupStream));
  NCCLCHECK(ncclProxyDestroy(comm));
  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(cudaSetDevice(savedDevice));

  TRACE(NCCL_INIT, "Destroyed comm %p rank %d", comm, rank);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %x", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  return commDestroy(comm);
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;

  return commDestroy(comm);
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));
  *asyncError = comm->fatalError;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}
