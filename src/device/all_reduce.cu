/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ring/all_reduce.cuh"
#include "tree/all_reduce.cuh"
#include "collnet/all_reduce.cuh"
#include "butterfly/all_reduce.cuh"
#include "butterfly2/all_reduce.cuh"
#include "butterfly_yz/all_reduce.cuh"
#include "mesh_cross/all_reduce.cuh"
#include "butterfly2d/all_reduce.cuh"
#include "ring2d/all_reduce.cuh"
#include "common.cuh"
#include "collectives.h"

IMPL_COLL_R(AllReduce);
