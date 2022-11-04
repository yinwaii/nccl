/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

#define MAXWIDTH_BTF 20
#define PREFIXLEN_BTF 15
#define STRLENGTH_BTF (PREFIXLEN_BTF+5*MAXWIDTH_BTF)
void dumpLine_btf(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH_BTF+1];
  line[STRLENGTH_BTF] = '\0';
  memset(line, ' ', STRLENGTH_BTF);
  strncpy(line, prefix, PREFIXLEN_BTF);
  for (int i=0; i<nranks && i<MAXWIDTH_BTF; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(NCCL_INIT,"%s", line);
}

ncclResult_t ncclBuildButterfly(int nwings, int* wings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nwings; r++) {
    char prefix[30];


    int current = rank;
    for (int i=0; i<nranks; i++) {
      wings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    sprintf(prefix, "Channel %02d/%02d : ", r, nwings);
    if (rank == 0) dumpLine_btf(wings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : butterfly %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (wings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}
