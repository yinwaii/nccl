/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "common.cuh"
#include "devcomm.h"
#include "primitives.cuh"

#include "ring/all_gather.cuh"

IMPL_COLL_C(AllGather);
