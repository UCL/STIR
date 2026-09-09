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
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"
#include <cuda_runtime.h>
#include <numeric>
#include <cstdio>

//the following is a pull operation
__global__ void forwardKernel(const float* __restrict__ in_im,
                             float* __restrict__ out_sino,
                             int3 dim)
{
    //1. define position in the image , x axial, z tangential, y is the view in terms of projection
    //    xz is the same of the detector. For image space x and z are facing the detector
    //    y is the integrating direction

    int det_x = blockIdx.x * blockDim.x + threadIdx.x;
    int det_y = blockIdx.y * blockDim.y + threadIdx.y;
    int det_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (det_x >= dim.x || det_y >= dim.y || det_z >= dim.z) {
        return;
    }
    int sino_idx = det_z * dim.x + (dim.x - 1 - det_x);//det_x; inversion of x to match SPECTUB

    // Sum voxel values along the y-axis (depth) for this detector pixel
//    int voxel_idx = det_y * dim.x * dim.z + det_z * dim.x + det_x;
    int voxel_idx = det_z*dim.x*dim.y + det_y*dim.x + det_x;
    //    out_sino[sino_idx] += in_im[voxel_idx];
    atomicAdd(&out_sino[sino_idx], in_im[voxel_idx]);

}


// the following is the adjoint operation (push)
__global__ void backwardKernel(const float* __restrict__ in_sino,
                             float* __restrict__ out_im,
                             int3 dim,
                             float3 spacing)
{
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idz < dim.z && idy < dim.y && idx < dim.x)
    {
//        if(idx==95 && idy==75 && idz==100)
//        {
//        printf("sino=%f\n",
//        in_sino[idz * dim.x + (dim.x - 1 - idx)]);
//        }
        out_im[(idz * dim.y + idy) * dim.x + idx] += in_sino[idz * dim.x + (dim.x - 1 - idx)] * spacing.x;//in_sino[idz * dim.y + idy] * spacing.x;

    }
}                          
