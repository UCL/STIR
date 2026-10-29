/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief implementations for cuda kernel for rotating projector with gaussian interpolation
  a la Wallis et al 1997,TMI, doi: 10.1109/42.552061.

  \author Daniel Deidda
  \author Hei Yin Jowett Chan
  \author Wei Huanzhe

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    Copyright (C) 2026, Convergent Imaging Solutions
    Copyright (C) 2026, King's College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include <cuda_runtime.h>

__global__ void
forwardKernel(float* sino, const float* in_image, const float* in_umap, int3 image_dim, float3 spacing, bool do_atten);

__global__ void
backwardKernel(float* image, const float* in_sino, const float* in_umap, int3 image_di, float3 spacing, bool do_atten);
