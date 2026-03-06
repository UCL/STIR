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
__global__ void rotateKernel_pull(const float* __restrict__ in_im,
                             float* __restrict__ out_im,
                             int3 dim,
                             float3 spacing,
                             float3 origin,
                             float angle_rad)
                             {
  // parallelise the operation across all image voxels
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  // check we are not outside the image
  if (i >= dim.x || j >= dim.y || k >= dim.z) return;

  // get the 1-dimensional index
  int idx = i + (j * dim.x) + (k * dim.x * dim.y);

  // move to centre of voxel and convert from voxel to image space
  float Xs_rel_x = (i + 0.5f) * spacing.x - origin.x;
  float Xs_rel_y = (j + 0.5f) * spacing.y - origin.y;
  float Xs_rel_z = (k + 0.5f) * spacing.z - origin.z;

  // perform the rotation by 'angle_rad'
  float X_rot_x = (Xs_rel_x * cosf(angle_rad)) - (Xs_rel_y * sinf(angle_rad));
  float X_rot_y = (Xs_rel_x * sinf(angle_rad)) + (Xs_rel_y * cosf(angle_rad));
  float X_rot_z = Xs_rel_z;

  // convert back to voxel space origin (corner of image) but keep image space dimensions
  float Xr_x = X_rot_x + origin.x;
  float Xr_y = X_rot_y + origin.y;
  float Xr_z = X_rot_z + origin.z;

  // now we go fully back to voxel space (but now it's been rotated)
  float p_r = Xr_x / spacing.x;
  float q_r = Xr_y / spacing.y;
  float r_r = Xr_z / spacing.z;

  // first loop is for finding the value of a gaussian kernel at
  // each nearest neighbour position and summing them all. This will allow normalisation
  // of each NN contribution in the next loop.
  float G = 0;  // accumulator variable
  float sigma = 1;  // gaussian kernel sigma

  for (int dr = -1; dr <= 1; dr++)  {
    int r = (int)roundf(r_r) + dr;  // nearest neighbour z coordinate in rotated voxel space
    float x_r = (r * spacing.z) - origin.z; // converted to image space

    for (int dq = -1; dq <= 1; dq++)  {
      int q = (int)roundf(q_r) + dq;
      float x_q = (q * spacing.y) - origin.y;

      for (int dp = -1; dp <= 1; dp++)  {
        int p = (int)roundf(p_r) + dp;
        float x_p = (p * spacing.x) - origin.x;

        // get distance between nearest neighbour and central voxel
        // both need to be in image space
        float delta_x = X_rot_x - x_p;
        float delta_y = X_rot_y - x_q;
        float delta_z = X_rot_z - x_r;

        // caulculate gaussian kernel 
        float g = expf(-1. * ((delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z)) / (2*sigma*sigma))
        G += g;
      };
    };
  };

  // loop again but this time actually fetch the image values
  float accumulation = 0;

  for (int dr = -1; dr <= 1; dr++)  {
    float r = dr + (int)roundf(r_r);
    float x_r = origin.z + (r * spacing.z);

    for (int dq = -1; dq <= 1; dq++)  {
      float q = dq + (int)roundf(q_r);
      float x_q = origin.y + (q * spacing.y);

      for (int dp = -1; dp <= 1; dp++)  {
        float p = dp + (int)roundf(p_r);
        float x_p = origin.x + (p * spacing.x);

        float delta_x = Xr_x - x_p;
        float delta_y = Xr_y - x_q;
        float delta_z = Xr_z - x_r;

        // get input image voxel index that we are at
        int i_idx = p + (q * dim.x) + (r * dim.x * dim.y);

        // calculate gaussian weighting for this voxel
        float g = expf(-1. * ((delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z)) / (2*sigma*sigma));

        // record the weighted value from this voxel
        accumulation += in_im[i_idx] * g / G;

      };
    };
  };

  // assign the pulled voxel values to a single voxel in the output image
  out_im[idx] = accumulation;

};

// the following is the adjoint operation (push)
__global__ void rotateKernel_push(const float* __restrict__ in_im,
                             float* __restrict__ out_im,
                             int3 dim,
                             float3 spacing,
                             float3 origin,
                             float angle_rad)
                             {
                              // parallelise the operation across all image voxels
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  int k = threadIdx.z + blockDim.z * blockIdx.z;

  // check we are not outside the image
  if (i >= dim.x || j >= dim.y || k >= dim.z) return;

  // get the 1-dimensional index
  int idx = i + (j * dim.x) + (k * dim.x * dim.y);

  // move to centre of voxel and convert from voxel to image space
  float Xs_rel_x = (i + 0.5f) * spacing.x - origin.x;
  float Xs_rel_y = (j + 0.5f) * spacing.y - origin.y;
  float Xs_rel_z = (k + 0.5f) * spacing.z - origin.z;

  // perform the rotation by 'angle_rad'
  float X_rot_x = (Xs_rel_x * cosf(angle_rad)) - (Xs_rel_y * sinf(angle_rad));
  float X_rot_y = (Xs_rel_x * sinf(angle_rad)) + (Xs_rel_y * cosf(angle_rad));
  float X_rot_z = Xs_rel_z;

  // convert back to voxel space origin (corner of image) but keep image space dimensions
  float Xr_x = X_rot_x + origin.x;
  float Xr_y = X_rot_y + origin.y;
  float Xr_z = X_rot_z + origin.z;

  // now we go fully back to voxel space (but now it's been rotated)
  float p_r = Xr_x / spacing.x;
  float q_r = Xr_y / spacing.y;
  float r_r = Xr_z / spacing.z;

  // first loop is for finding the value of a gaussian kernel at
  // each nearest neighbour position and summing them all. This will allow normalisation
  // of each NN contribution in the next loop.
  float G = 0;  // accumulator variable
  float sigma = 1;  // gaussian kernel sigma

  for (int dr = -1; dr <= 1; dr++)  {
    int r = (int)roundf(r_r) + dr;  // nearest neighbour z coordinate in rotated voxel space
    float x_r = (r * spacing.z) - origin.z; // converted to image space

    for (int dq = -1; dq <= 1; dq++)  {
      int q = (int)roundf(q_r) + dq;
      float x_q = (q * spacing.y) - origin.y;

      for (int dp = -1; dp <= 1; dp++)  {
        int p = (int)roundf(p_r) + dp;
        float x_p = (p * spacing.x) - origin.x;

        // get distance between nearest neighbour and central voxel
        // both need to be in image space
        float delta_x = X_rot_x - x_p;
        float delta_y = X_rot_y - x_q;
        float delta_z = X_rot_z - x_r;

        // caulculate gaussian kernel 
        float g = expf(-1. * ((delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z)) / (2*sigma*sigma))
        G += g;
      };
    };
  };

  // loop again but this time actually fetch the image values
  float accumulation = 0;

  for (int dr = -1; dr <= 1; dr++)  {
    float r = dr + (int)roundf(r_r);
    float x_r = origin.z + (r * spacing.z);

    for (int dq = -1; dq <= 1; dq++)  {
      float q = dq + (int)roundf(q_r);
      float x_q = origin.y + (q * spacing.y);

      for (int dp = -1; dp <= 1; dp++)  {
        float p = dp + (int)roundf(p_r);
        float x_p = origin.x + (p * spacing.x);

        float delta_x = Xr_x - x_p;
        float delta_y = Xr_y - x_q;
        float delta_z = Xr_z - x_r;

        // get input image voxel index that we are at
        int i_idx = p + (q * dim.x) + (r * dim.x * dim.y);

        // Pushing the weighted counts to NN
        float g = expf(-1. * ((delta_x*delta_x) + (delta_y*delta_y) + (delta_z*delta_z)) / (2*sigma*sigma));
        atomicAdd(&out_im[i_idx], in_im[idx]*g/G); 
                              
         };
    };
  };
}
