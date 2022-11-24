/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.cuh"
#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"

#include "ring/all_reduce.cuh"
#include "tree/all_reduce.cuh"
#include "collnet/all_reduce.cuh"
#include "butterfly/all_reduce.cuh"
#include "butterfly/all_reduce_2.cuh"
#include "butterfly/all_reduce_yz.cuh"

IMPL_COLL_R(AllReduce);
