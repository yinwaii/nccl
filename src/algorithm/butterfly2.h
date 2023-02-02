#ifndef __BUTTERFLY2_H__
#define __BUTTERFLY2_H__
#include "base.h"
#include "butterfly.h"
#include "comm.h"
#include "info.h"

using ncclTopoButterfly2 = ncclTopoButterfly;

class ncclEnqueueButterfly2 : public ncclEnqueueBase {
private:
  enum Patterns { ncclPatternButterfly, ncclPatternHalfDoubling };
  int getNsteps(struct ncclProxyArgs *args, struct ncclInfo *info,
                size_t size) const;

public:
  ncclEnqueueButterfly2() : ncclEnqueueBase("Butterfly") {}
  ncclResult_t getPattern(int coll, int *pattern) const;
  ncclResult_t enqueueRedirect(struct ncclInfo *info) const;
  ncclResult_t enqueueLoopInfo(struct ncclInfo *info) const;
  ncclResult_t proxySaveColl(struct ncclProxyArgs *args,
                             struct ncclInfo *info) const;
};

using ncclTuningButterfly2 = ncclTuningButterfly;

#endif