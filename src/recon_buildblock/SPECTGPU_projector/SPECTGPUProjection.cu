//
//
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

#include "stir/cuda_utilities.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"
#include <cuda_runtime.h>
#include <numeric>
#include <cstdio>

// the following is a pull operation
__global__ void
forwardKernel(float* __restrict__ out_sino,
              const float* __restrict__ in_im,
              const float* __restrict__ in_umap,
              int3 dim,
              float3 spacing,
              bool do_atten)
{
  // 1. define position in the image , x axial, z tangential, y is the view in terms of projection
  //     xz is the same of the detector. For image space x and z are facing the detector
  //     y is the integrating direction

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  float atten;

  if (idx >= dim.x || idy >= dim.y || idz >= dim.z)
    {
      return;
    }
  int sino_idx = idz * dim.x + (dim.x - 1 - idx); // idx; inversion of x to match SPECTUB

  // Sum voxel values along the y-axis (depth) for this detector pixel
  //    int voxel_idx = idy * dim.x * dim.z + idz * dim.x + idx;
  int voxel_idx = idz * dim.x * dim.y + idy * dim.x + idx;

  //    out_sino[sino_idx] += in_im[voxel_idx];
  if (do_atten)
    {
      float mu_integral = 0.0f;

      for (int y = idy; y < dim.y; y++)
        {
          int id = idz * dim.x * dim.y + y * dim.x + idx;

          float voxel_length_cm = spacing.y * 0.1f;

          mu_integral += in_umap[id] * voxel_length_cm; // need to be in, cm/cm, same units
        }

      atten = expf(-mu_integral);
      //        atten = expf(-in_umap[voxel_idx]*spacing.y);
    }
  else
    atten = 1;

  atomicAdd(&out_sino[sino_idx], in_im[voxel_idx] * atten);
}

// the following is the adjoint operation (push)
__global__ void
backwardKernel(float* __restrict__ out_im,
               const float* __restrict__ in_sino,
               const float* __restrict__ in_umap,
               int3 dim,
               float3 spacing,
               bool do_atten)
{
  int idz = blockIdx.z * blockDim.z + threadIdx.z;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float atten;

  if (idz < dim.z && idy < dim.y && idx < dim.x)
    {
      //        if(idx==95 && idy==75 && idz==100)
      //        {
      //        printf("sino=%f\n",
      //        in_sino[idz * dim.x + (dim.x - 1 - idx)]);
      //        }
      if (do_atten)
        {
          float mu_integral = 0.0f;

          for (int y = idy; y < dim.y; y++)
            {
              int id = idz * dim.x * dim.y + y * dim.x + idx;

              float voxel_length_cm = spacing.y * 0.1f;
              mu_integral += in_umap[id] * voxel_length_cm;
            }

          atten = expf(-mu_integral);
          //            if(idx==75 && idy==75 && idz==90)
          //            {
          //                printf("mu=%f atten=%f\n",
          //                       mu_integral,
          //                       atten);
          //            }
        }
      else
        atten = 1;
      out_im[(idz * dim.y + idy) * dim.x + idx]
          += atten * in_sino[idz * dim.x + (dim.x - 1 - idx)]; // in_sino[idz * dim.y + idy] * spacing.x;
    }
}
