/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief implementations for cuda kernel of depth dependendent PSF

  \author Daniel Deidda

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/cuda_utilities.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUPSFGaussian.h"
#include <cuda_runtime.h>
#include <numeric>
#include <cstdio>

// the following is a pull operation
__global__ void
GaussianConvolutionKernel_pull(
    float* __restrict__ out_im, const float* __restrict__ in_im, int3 dim, float3 spacing, float sigma0, float slope)
{
  // parallelise the operation across all image voxels
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  // check we are not outside the image
  if (i >= dim.x || j >= dim.y || k >= dim.z)
    return;

  // get the 1-dimensional index
  int id = i + (j * dim.x) + (k * dim.x * dim.y);

  // calculate sigma dependent on depth (y)
  float d = (dim.y - 1 - j) * spacing.y * 0.1f; // in cm
  float sigma = sigma0 + slope * d;
  float G = 0;
  float accumulation = 0;
  //    following assumes spacingx=spacinz could be needed to
  float sigma_x_vox = sigma / (spacing.x * 0.1f);
  float sigma_z_vox = sigma / (spacing.z * 0.1f);

  //  definition of 3*sigma radius;
  //  we could even do the number_of_sigmas like in SPECTUB
  int Rz = max(2, (int)ceilf(3.f * sigma_z_vox));
  int Rx = max(2, (int)ceilf(3.f * sigma_x_vox));

  for (int dz = -Rz; dz <= Rz; dz++)
    {
      int z = dz + k; // nearest neighbour z coordinate

      for (int dx = -Rx; dx <= Rx; dx++)
        {
          int x = dx + i;

          if (x < 0 || x >= dim.x || z < 0 || z >= dim.z)
            continue;

          // caulculate gaussian kernel
          float dx_cm = dx * spacing.x * 0.1f;
          float dz_cm = dz * spacing.z * 0.1f;

          float g = expf(-(dx_cm * dx_cm + dz_cm * dz_cm) / (2.f * sigma * sigma));
          G += g;
        };
    };

  for (int dz = -Rz; dz <= Rz; dz++)
    {
      int z = dz + k; // nearest neighbour z coordinate

      for (int dx = -Rx; dx <= Rx; dx++)
        {
          int x = dx + i;
          if (x < 0 || x >= dim.x || z < 0 || z >= dim.z)
            continue;

          int id_n = z * dim.x * dim.y + j * dim.x + x;
          // Pushing the weighted counts to NN
          float dx_cm = dx * spacing.x * 0.1f;
          float dz_cm = dz * spacing.z * 0.1f;

          float g = expf(-(dx_cm * dx_cm + dz_cm * dz_cm) / (2.f * sigma * sigma));

          accumulation += in_im[id_n] * g / G;
        };
    };
  out_im[id] = accumulation;
};

// the following is the adjoint operation (push)
__global__ void
GaussianConvolutionKernel_push(
    float* __restrict__ out_im, const float* __restrict__ in_im, int3 dim, float3 spacing, float sigma0, float slope)
{
  // parallelise the operation across all image voxels
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  // check we are not outside the image
  if (i >= dim.x || j >= dim.y || k >= dim.z)
    return;

  // get the 1-dimensional index
  int id = i + (j * dim.x) + (k * dim.x * dim.y);

  // calculate sigma dependent on depth (y)
  float d = (dim.y - 1 - j) * spacing.y * 0.1f; // in cm
  float sigma = sigma0 + slope * d;
  float G = 0;

  float sigma_x_vox = sigma / (spacing.x * 0.1f);
  float sigma_z_vox = sigma / (spacing.z * 0.1f);

  //  definition of 3*sigma radius;
  //  we could even do the number_of_sigmas like in SPECTUB
  int Rz = max(2, (int)ceilf(3.f * sigma_z_vox));
  int Rx = max(2, (int)ceilf(3.f * sigma_x_vox));

  for (int dz = -Rz; dz <= Rz; dz++)
    {
      int z = dz + k; // nearest neighbour z coordinate

      for (int dx = -Rx; dx <= Rx; dx++)
        {
          int x = dx + i;

          if (x < 0 || x >= dim.x || z < 0 || z >= dim.z)
            continue;

          // caulculate gaussian kernel
          float dx_cm = dx * spacing.x * 0.1f;
          float dz_cm = dz * spacing.z * 0.1f;

          float g = expf(-(dx_cm * dx_cm + dz_cm * dz_cm) / (2.f * sigma * sigma));
          G += g;
        };
    };

  for (int dz = -Rz; dz <= Rz; dz++)
    {
      int z = dz + k; // nearest neighbour z coordinate

      for (int dx = -Rx; dx <= Rx; dx++)
        {
          int x = dx + i;
          if (x < 0 || x >= dim.x || z < 0 || z >= dim.z)
            continue;

          int id_n = z * dim.x * dim.y + j * dim.x + x;
          // Pushing the weighted counts to NN
          float dx_cm = dx * spacing.x * 0.1f;
          float dz_cm = dz * spacing.z * 0.1f;

          float g = expf(-(dx_cm * dx_cm + dz_cm * dz_cm) / (2.f * sigma * sigma));

          atomicAdd(&out_im[id_n], in_im[id] * g / G);
        };
    };
}
