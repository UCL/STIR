//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief implementations for cuda kernel for rotating projector with gaussian interpolation 
  a la Wallis et al 1997,TMI, doi: 10.1109/42.552061.

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/cuda_utilities.h"
#include <cuda_runtime.h>
#include <numeric>


//the following is a pull operation
__global__ void forwardKernel(const float* __restrict__ in_im,
                             float* __restrict__ out_sino,
                             int3 dim,
                             float3 spacing,
                             float3 origin,
                             float angle_rad)
                             {


                             }

// the following is the adjoint operation (push)
__global__ void backwardKernel(const float* __restrict__ in_sino,
                             float* __restrict__ out_im,
                             int3 dim,
                             float3 spacing,
                             float3 origin,
                             float angle_rad)
                             {


                             }                             
