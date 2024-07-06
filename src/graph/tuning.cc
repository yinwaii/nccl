/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "devcomm.h"
#include "comm.h"
#include "tuning.h"
#include "interface.h"
#include "topo.h"

NCCL_PARAM(Nthreads, "NTHREADS", -2);
NCCL_PARAM(Ll128Nthreads, "LL128_NTHREADS", -2);

int getNthreads(const char* name, int env, int min, int max, int def) {
  int nt = env;
  if (nt > 0) {
    if (nt % WARP_SIZE != 0) {
      WARN("Invalid %s %d (must be a multiple of %d)", name, nt, WARP_SIZE);
      nt = max;
    } else if (nt > max) {
      WARN("Invalid %s %d (maximum %d).", name, nt, max);
      nt = max;
    } else if (nt < min) {
      WARN("Invalid %s %d (minimum %d).", name, nt, min);
      nt = min;
     }
  } else {
    nt = def;
  }
  return nt;
}

ncclResult_t parseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

ncclResult_t ncclTopoEnable(struct ncclComm *comm) {
  // Protocols/Algorithms enable/disable, and user overrides.
  // All are enabled except ll128 which is enabled by default only in certain cases.
  comm->protoEnable[NCCL_PROTO_LL] = 1;
  comm->protoEnable[NCCL_PROTO_LL128] = 2;
  comm->protoEnable[NCCL_PROTO_SIMPLE] = 1;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) 
    comm->algoEnable[a] = 1;

  const char *protoStr = getenv("NCCL_PROTO");
  if (protoStr) {
    INFO(NCCL_ENV, "NCCL_PROTO set by environment to %s", protoStr);
    NCCLCHECK(parseList(protoStr, ncclProtoStr, NCCL_NUM_PROTOCOLS, comm->protoEnable));
  }
  const char *algoStr = getenv("NCCL_ALGO");
  if (algoStr) {
    INFO(NCCL_ENV, "NCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(parseList(algoStr, ncclAlgoStr, NCCL_NUM_ALGORITHMS, comm->algoEnable));
  }
  // Disable CollNet if it is not supported
  // if (comm->collNetSupport == 0) {
  //   // If user has hard set NCCL_ALGO=COLLNET, ignore it
  //   bool onlyCollnet = true;
  //   for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
  //     if (comm->algoEnable[a] != 0 && a != NCCL_ALGO_COLLNET) {
  //       onlyCollnet = false;
  //       break;
  //     }
  //   }
  //   if (onlyCollnet) {
  //     for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) 
  //       comm->algoEnable[a] = 1;
  //     if (comm->rank == 0) WARN("CollNet is not supported or fails to initialize, ignoring NCCL_ALGO=COLLNET");
  //   }
  //   comm->algoEnable[NCCL_ALGO_COLLNET] = 0;
  // }
  return ncclSuccess;
}

ncclResult_t ncclTopoTuneEnable(struct ncclComm *comm, int minCompCap, int maxCompCap, AlgoInfo<ncclTuningAlgo> algos) {
  for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int pEnable = comm->protoEnable[p];
    if (pEnable == 2 && p == NCCL_PROTO_LL128) {
      // Enable LL128 by default only on Volta/Ampere+NVLink. Other cases are not tested and may cause silent data corruption.
      pEnable = (algos[a]->topo->graph.typeInter <= PATH_PXB) && algos[a]->topo->graph.typeIntra <= PATH_NVL &&
        ((minCompCap == 70 && maxCompCap == 70) || (minCompCap == 80 && maxCompCap == 80)) ? 1 : 0;
    }
    if (pEnable == 0) comm->tuning[a].bandwidths[c][p] = 0;
    // Only disable algo for Allreduce since others only have one
    if (comm->algoEnable[a] == 0) comm->tuning[a].bandwidths[c][p] = 0;
  }

  return ncclSuccess;
}

ncclResult_t ncclTuningDumpLatBw(struct ncclComm *comm) {
  if (comm->rank == 0) {
    char line[1024];
    sprintf(line, "Latency/AlgBw |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %7s/%6s |", ncclAlgoStr[a], ncclProtoStr[p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    sprintf(line, " Max NThreads |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        sprintf(line+strlen(line), " %14d |", comm->tuning[a].maxThreads[p]);
      }
    }
    INFO(NCCL_TUNING, "%s", line);
    for (int c=0; c<NCCL_NUM_FUNCTIONS; c++) {
      sprintf(line, "%13s |", ncclFuncStr[c]);
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
        for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
          sprintf(line+strlen(line), "%8.1f/%6.1f |", comm->tuning[a].latencies[c][p], comm->tuning[a].bandwidths[c][p]);
        }
      }
      INFO(NCCL_TUNING, "%s", line);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTuningLoadThresholds(struct ncclComm *comm) {
  // Override defaults with user env
  char* str = getenv("NCCL_THREAD_THRESHOLDS");
  if (str) {
    INFO(NCCL_ENV, "NCCL_THREAD_THRESHOLDS set by environment to %s", str);
    ssize_t t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {{ -2, -2, -2 }, { -2, -2, -2}};
    sscanf(str, "%ld %ld %ld %ld %ld %ld", t[0], t[0]+1, t[0]+2, t[1], t[1]+1, t[1]+2);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (t[a][p] >= 0) comm->tuning[a].threadThresholds[p] = t[a][p];
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTuningDumpThresholds(struct ncclComm *comm) {
  char line[1024];
  sprintf(line, "threadThresholds ");
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
  {
    if (a > 0) sprintf(line + strlen(line), " | ");
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
    {
      if (p > 0) sprintf(line + strlen(line), "/");
      sprintf(line + strlen(line), "%ld", comm->tuning[a].threadThresholds[p]);
    }
  }
  INFO(NCCL_INIT, "%s", line);
  return ncclSuccess;
}

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, AlgoInfo<ncclTopoAlgo> topo) {
  AlgoInfo<ncclTuningAlgo> algos = ncclTuningAlgos(comm, topo);
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    NCCLCHECK(algos[a]->tuningMaxThreads(a));
  }

  if (comm->nRanks <= 1) return ncclSuccess;

  int compCap80 = minCompCap == 80 && maxCompCap == 80 ? 1 : 0;

  for (int coll=0; coll<NCCL_NUM_FUNCTIONS; coll++) {
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      int pattern;
      NCCLCHECK(ncclAlgos[a]->getPattern(coll, &pattern));
      if (pattern == -1) continue;
      NCCLCHECK(algos[a]->tuningBw(coll, a, compCap80));
      NCCLCHECK(algos[a]->tuningLat(coll, a));
    }
  }

  NCCLCHECK(ncclTopoTuneEnable(comm, minCompCap, maxCompCap, algos));

  NCCLCHECK(ncclTuningDumpLatBw(comm));

  // Set per-thread amount of work before we increase nThreads and nChannels
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    NCCLCHECK(algos[a]->tuningThresholds(a));
  }

  NCCLCHECK(ncclTuningLoadThresholds(comm));
  NCCLCHECK(ncclTuningDumpThresholds(comm));

  return ncclSuccess;
}

ncclResult_t ncclTuningAlgoTime(struct ncclInfo* info, int algorithm, int protocol, float* time) {
  float bw = info->comm->tuning[algorithm].bandwidths[info->coll][protocol];
  float lat = info->comm->tuning[algorithm].latencies[info->coll][protocol];
  if (bw == 0) {
    *time = -1.0; return ncclSuccess;
  }
  *time = lat + (info->nBytes) / (1000 * bw);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetAlgoTime(struct ncclInfo* info, int algorithm, int protocol, float* time) {
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++)
    NCCLCHECK(ncclAlgos[a]->tuningAlgoTime(info, algorithm, protocol, time));
  // INFO(NCCL_TUNING, "Algorithm time: algo %s, proto %s, time %lf", ncclAlgoStr[algorithm], ncclProtoStr[protocol], *time);
  return ncclSuccess;
}
