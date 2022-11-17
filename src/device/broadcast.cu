/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.cuh"
#include "devcomm.h"
#include "primitives.cuh"
#include "collectives.h"

#include "ring/broadcast.cuh"

IMPL_COLL_C(Broadcast);
