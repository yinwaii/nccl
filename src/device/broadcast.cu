/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "broadcast.cuh"
#include "common.cuh"
#include "collectives.h"

IMPL_COLL_C(ncclBroadcast, ncclCollBroadcast);
