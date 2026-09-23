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

#include <cuda_runtime.h>

__global__ void
GaussianConvolutionKernel_pull(float* out_im, const float* in_im, int3 image_dim, float3 spacing, float sigma0, float slope);

__global__ void
GaussianConvolutionKernel_push(float* out_im, const float* in_im, int3 image_dim, float3 spacing, float sigma0, float slope);
