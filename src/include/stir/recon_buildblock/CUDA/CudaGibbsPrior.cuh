//
//
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier:See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Implementation of class stir::CudaGibbsPrior

  \author Matteo Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_CUDA_CudaGibbsPrior_CUH__
#define __stir_recon_buildblock_CUDA_CudaGibbsPrior_CUH__

#include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"

#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/cuda_utilities.h"
#include "stir/info.h"
#include "stir/warning.h"
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>

START_NAMESPACE_STIR

// CUDA kernels for CudaGibbsPrior

// typedef for the value (array) in the CUDA kernel
// We need double for numerical accuracy according to test_priors.cxx
// typedef double value_type;

// Generic, architecture-compatible atomicAdd for double and convertible types
template <typename elemT>
__device__ inline double
atomicAddGeneric(double* address, elemT val)
{
  double dval = static_cast<double>(val);
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, dval);
#else
  // Use atomicCAS for double-precision atomicAdd on older architectures
  unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do
    {
      assumed = old;
      double updated = __longlong_as_double(assumed) + dval;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(updated));
  } while (assumed != old);

  return __longlong_as_double(old);
#endif
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Value_Kernel(double* output,
             const elemT* __restrict__ input_image,
             const float* __restrict__ weights,
             const elemT* __restrict__ kappa,
             const bool do_kappa,
             const int3 d_Image_dim,
             const int3 d_Image_max_indices,
             const int3 d_Image_min_indices,
             const int3 d_weight_max_indices,
             const int3 d_weight_min_indices,
             const CudaPotentialT cuda_potential_fun)
{
  extern __shared__ elemT block_sum_shared[];

  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  int thread_in_block = threadIdx.z * blockDim.y * blockDim.x +
                        threadIdx.y * blockDim.x + 
                        threadIdx.x;
  // Initialize shared memory for block sum
  block_sum_shared[thread_in_block] = elemT(0);
  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;
  // Convert thread indices to actual image coordinates
  const int3 Image_coord = { d_Image_min_indices.x + thread_id_x, 
                                 d_Image_min_indices.y + thread_id_y,
                                 d_Image_min_indices.z + thread_id_z };
  // Get the index of the voxel (memory access is still 0-based)
  const int centerIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + 
                          thread_id_y * d_Image_dim.x + 
                          thread_id_x;

  const elemT val_center = input_image[centerIndex];

  elemT sum = 0.0;
  //Boundary conditions for the neighbourhood
  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;

  for (int dz = min_dz; dz <= max_dz; dz++)
    for (int dy = min_dy; dy <= max_dy; dy++)
      for (int dx = min_dx; dx <= max_dx; dx++)
        {
          // Linearized index for the neighbour voxel
          const int neighbourIndex = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x +
                                     (thread_id_y + dy) * d_Image_dim.x +
                                     (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex   = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x +
                                     (dy - d_weight_min_indices.y) * weights_size_x + 
                                     (dx - d_weight_min_indices.x);

          const elemT val_neigh = input_image[neighbourIndex];
          elemT current = weights[weightsIndex] *
                          cuda_potential_fun.value(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            current *= kappa[centerIndex] * kappa[neighbourIndex];

          sum += current;
        }
  block_sum_shared[thread_in_block] = sum;
  __syncthreads();

  // Parallel reduction in shared memory
  int block_threads = blockDim.x * blockDim.y * blockDim.z;
  for (int stride = block_threads / 2; stride > 0; stride /= 2)
    {
      if (thread_in_block < stride)
        block_sum_shared[thread_in_block] += block_sum_shared[thread_in_block + stride];
      __syncthreads();
    }

  // // One thread per block performs final atomic add in double
  if (thread_in_block == 0)
      atomicAddGeneric(output, block_sum_shared[0]);    
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Gradient_Kernel(elemT* grad,
                const elemT* image,
                const float* weights,
                const elemT* kappa,
                const bool do_kappa,
                const float penalisation_factor,
                const int3 d_Image_dim,
                const int3 d_Image_max_indices,
                const int3 d_Image_min_indices,
                const int3 d_weight_max_indices,
                const int3 d_weight_min_indices,
                const CudaPotentialT cuda_potential_fun)
{
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;
  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;
  // Convert thread indices to actual image coordinates
  const int3 Image_coord = {d_Image_min_indices.x + thread_id_x, 
                             d_Image_min_indices.y + thread_id_y,
                             d_Image_min_indices.z + thread_id_z};
  // Get the index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + 
                         thread_id_y * d_Image_dim.x +
                         thread_id_x;

  const elemT val_center = image[inputIndex];
  //Boundary conditions for the neighbourhood
  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;

  // Define variable to sum over the neighbourhood
  elemT sum = 0.0;

  for (int dz = min_dz; dz <= max_dz; dz++)
      for (int dy = min_dy; dy <= max_dy; dy++)
          for (int dx = min_dx; dx <= max_dx; dx++)
            {
              //Linearized index for the neighbour voxel
              const int neighbourIndex = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x +
                                         (thread_id_y + dy) * d_Image_dim.x +
                                         (thread_id_x + dx);
              //Linearized index for the weight
              const int weightsIndex   = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x +
                                         (dy - d_weight_min_indices.y) * weights_size_x +
                                         (dx - d_weight_min_indices.x);
              //Neighbour voxel contribution
              const elemT val_neigh    = image[neighbourIndex];
              elemT current            = weights[weightsIndex] *
                                         cuda_potential_fun.derivative_10(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

              if (do_kappa)  
                current *= kappa[inputIndex] * kappa[neighbourIndex];
              
              sum += current;
            }
  grad[inputIndex] = sum * 2.0 * penalisation_factor;
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Gradient_exp_Kernel(elemT* grad,
                const elemT* image,
                const float* weights,
                const elemT* kappa,
                const bool do_kappa,
                const float penalisation_factor,
                const int3 d_Image_dim,
                const int3 d_Image_max_indices,
                const int3 d_Image_min_indices,
                const int3 d_weight_max_indices,
                const int3 d_weight_min_indices,
                const CudaPotentialT cuda_potential_fun)
{ 
  int tile_dim_x = blockDim.x + 2;
  int tile_dim_y = blockDim.y + 2;
  int tile_dim_z = blockDim.z + 2;
  #define TILE_INDEX(z, y, x) (((z) * tile_dim_y * tile_dim_x) + ((y) * tile_dim_x) + (x)) 
  extern __shared__ elemT tile[];

  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;
  int z = blockIdx.z * blockDim.z + tz;

  // Fill shared memory tile (center)
  tile[TILE_INDEX(tz+1, ty+1, tx+1)] = get_voxel(image, x, y, z, d_Image_dim);

  // Fill halo faces
  if (tx == 0)
    tile[TILE_INDEX(tz+1, ty+1, 0)] = get_voxel(image, x-1, y, z, d_Image_dim);
  if (tx == blockDim.x-1)
    tile[TILE_INDEX(tz+1, ty+1, blockDim.x+1)] = get_voxel(image, x+1, y, z, d_Image_dim);
  if (ty == 0)
    tile[TILE_INDEX(tz+1, 0, tx+1)] = get_voxel(image, x, y-1, z, d_Image_dim);
  if (ty == blockDim.y-1)
    tile[TILE_INDEX(tz+1, blockDim.y+1, tx+1)] = get_voxel(image, x, y+1, z, d_Image_dim);
  if (tz == 0)
    tile[TILE_INDEX(0, ty+1, tx+1)] = get_voxel(image, x, y, z-1, d_Image_dim);
  if (tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, ty+1, tx+1)] = get_voxel(image, x, y, z+1, d_Image_dim);

  // X-Y edges
  if (tz == 0 && ty == 0)
    tile[TILE_INDEX(0, 0, tx+1)] = get_voxel(image, x, y-1, z-1, d_Image_dim);
  if (tz == 0 && ty == blockDim.y-1)
    tile[TILE_INDEX(0, blockDim.y+1, tx+1)] = get_voxel(image, x, y+1, z-1, d_Image_dim);
  if (tz == blockDim.z-1 && ty == 0)
    tile[TILE_INDEX(blockDim.z+1, 0, tx+1)] = get_voxel(image, x, y-1, z+1, d_Image_dim);
  if (tz == blockDim.z-1 && ty == blockDim.y-1)
    tile[TILE_INDEX(blockDim.z+1, blockDim.y+1, tx+1)] = get_voxel(image, x, y+1, z+1, d_Image_dim);

  // X-Z edges
  if (ty == 0 && tx == 0)
    tile[TILE_INDEX(tz+1, 0, 0)] = get_voxel(image, x-1, y-1, z, d_Image_dim);
  if (ty == 0 && tx == blockDim.x-1)
    tile[TILE_INDEX(tz+1, 0, blockDim.x+1)] = get_voxel(image, x+1, y-1, z, d_Image_dim);
  if (ty == blockDim.y-1 && tx == 0)
    tile[TILE_INDEX(tz+1, blockDim.y+1, 0)] = get_voxel(image, x-1, y+1, z, d_Image_dim);
  if (ty == blockDim.y-1 && tx == blockDim.x-1)
    tile[TILE_INDEX(tz+1, blockDim.y+1, blockDim.x+1)] = get_voxel(image, x+1, y+1, z, d_Image_dim);

  // Y-Z edges
  if (tx == 0 && tz == 0)
    tile[TILE_INDEX(0, ty+1, 0)] = get_voxel(image, x-1, y, z-1, d_Image_dim);
  if (tx == 0 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, ty+1, 0)] = get_voxel(image, x-1, y, z+1, d_Image_dim);
  if (tx == blockDim.x-1 && tz == 0)
    tile[TILE_INDEX(0, ty+1, blockDim.x+1)] = get_voxel(image, x+1, y, z-1, d_Image_dim);
  if (tx == blockDim.x-1 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, ty+1, blockDim.x+1)] = get_voxel(image, x+1, y, z+1, d_Image_dim);

  // 8 corners
  if (tx == 0 && ty == 0 && tz == 0)
    tile[TILE_INDEX(0, 0, 0)] = get_voxel(image, x-1, y-1, z-1, d_Image_dim);
  if (tx == 0 && ty == 0 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, 0, 0)] = get_voxel(image, x-1, y-1, z+1, d_Image_dim);
  if (tx == 0 && ty == blockDim.y-1 && tz == 0)
    tile[TILE_INDEX(0, blockDim.y+1, 0)] = get_voxel(image, x-1, y+1, z-1, d_Image_dim);
  if (tx == 0 && ty == blockDim.y-1 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, blockDim.y+1, 0)] = get_voxel(image, x-1, y+1, z+1, d_Image_dim);

  if (tx == blockDim.x-1 && ty == 0 && tz == 0)
    tile[TILE_INDEX(0, 0, blockDim.x+1)] = get_voxel(image, x+1, y-1, z-1, d_Image_dim);
  if (tx == blockDim.x-1 && ty == 0 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, 0, blockDim.x+1)] = get_voxel(image, x+1, y-1, z+1, d_Image_dim);
  if (tx == blockDim.x-1 && ty == blockDim.y-1 && tz == 0)
    tile[TILE_INDEX(0, blockDim.y+1, blockDim.x+1)] = get_voxel(image, x+1, y+1, z-1, d_Image_dim);
  if (tx == blockDim.x-1 && ty == blockDim.y-1 && tz == blockDim.z-1)
    tile[TILE_INDEX(blockDim.z+1, blockDim.y+1, blockDim.x+1)] = get_voxel(image, x+1, y+1, z+1, d_Image_dim);

  __syncthreads();

  // Check if the thread is within the image dimensions
  if (z >= d_Image_dim.z || y >= d_Image_dim.y || x >= d_Image_dim.x)
    return;
  
  // Local coordinates in the tile
  int lz = tz + 1, ly = ty + 1, lx = tx + 1;

  // Convert thread indices to actual image coordinates
  const int3 Image_coord = {d_Image_min_indices.x + x, 
                             d_Image_min_indices.y + y,
                             d_Image_min_indices.z + z};
  // Get the index of the voxel (memory access is still 0-based)
  const int inputIndex = z * d_Image_dim.y * d_Image_dim.x + 
                         y * d_Image_dim.x +
                         x;
  
  const elemT val_center = tile[TILE_INDEX(lz, ly, lx)];

  // Full stencil window (no boundary checks needed)
  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;

  elemT sum = 0.0;

  for (int dz = min_dz; dz <= max_dz; dz++)
    for (int dy = min_dy; dy <= max_dy; dy++)
      for (int dx = min_dx; dx <= max_dx; dx++)
      {
          elemT val_neigh = tile[TILE_INDEX(lz + dz, ly + dy, lx + dx)];

          const int weightsIndex   = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x +
                                     (dy - d_weight_min_indices.y) * weights_size_x +
                                     (dx - d_weight_min_indices.x);

          elemT current = weights[weightsIndex] *
                          cuda_potential_fun.derivative_10(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)  
              current *= kappa[inputIndex] * kappa[inputIndex /* or neighbor, if needed */];

          sum += current;
      }
  grad[inputIndex] = sum * 2.0 * penalisation_factor;
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Gradient_dot_input_Kernel(double* output,
             const elemT* __restrict__ input,
             const elemT* __restrict__ current_image,
             const float* __restrict__ weights,
             const elemT* __restrict__ kappa,
             const bool do_kappa,
             const int3 d_Image_dim,
             const int3 d_Image_max_indices,
             const int3 d_Image_min_indices,
             const int3 d_weight_max_indices,
             const int3 d_weight_min_indices,
             const CudaPotentialT cuda_potential_fun)
{
  extern __shared__ elemT block_sum_shared[];

  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  int thread_in_block = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
  block_sum_shared[thread_in_block] = elemT(0);

  // Check if the thread is within the image dimensions
  // Note: only image_min_z is 0-based indices
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;

  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };

  // Get the index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + thread_id_y * d_Image_dim.x + thread_id_x;
  const elemT val_center = current_image[inputIndex];

  // Define the sum variable
  elemT sum = 0.0;
  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;



  for (int dz = min_dz; dz <= max_dz; dz++)
    for (int dy = min_dy; dy <= max_dy; dy++)
      for (int dx = min_dx; dx <= max_dx; dx++)
        {
          //Linearized index for the neighbour voxel
          const int neighbourIndex = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x +
                                      (thread_id_y + dy) * d_Image_dim.x +
                                      (thread_id_x + dx);
          //Linearized index for the weight
          const int weightsIndex   = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x +
                                      (dy - d_weight_min_indices.y) * weights_size_x +
                                      (dx - d_weight_min_indices.x);
          //Neighbour voxel contribution
          const elemT val_neigh    = current_image[neighbourIndex];

          elemT current = weights[weightsIndex]
                          * cuda_potential_fun.derivative_10(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            current *= kappa[inputIndex] * kappa[neighbourIndex];

          sum += current;
        }
  block_sum_shared[thread_in_block] = 2 * sum * input[inputIndex];
  __syncthreads();

  // Parallel reduction in shared memory
  int block_threads = blockDim.x * blockDim.y * blockDim.z;
  for (int stride = block_threads / 2; stride > 0; stride /= 2)
    {
      if (thread_in_block < stride)
        block_sum_shared[thread_in_block] += block_sum_shared[thread_in_block + stride];
      __syncthreads();
    }

  // // One thread per block performs final atomic add in double
  // int block_id = blockIdx.z * gridDim.y * gridDim.x +
  //                blockIdx.y * gridDim.x +
  //                blockIdx.x;

  if (thread_in_block == 0)
      atomicAddGeneric(output, block_sum_shared[0]);
      // output[block_id] = static_cast<double>(block_sum_shared[0])  ; // Store the final sum in the output
    
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Hessian_diagonal_Kernel(elemT* Hessian_diag,
                const elemT* image,
                const float* weights,
                const elemT* kappa,
                const bool do_kappa,
                const float penalisation_factor,
                const int3 d_Image_dim,
                const int3 d_Image_max_indices,
                const int3 d_Image_min_indices,
                const int3 d_weight_max_indices,
                const int3 d_weight_min_indices,
                const CudaPotentialT cuda_potential_fun)
{
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if the thread is within the image dimensions
  // Note: only image_min_z is 0-based indices
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;

  // Convert thread indices to actual image coordinates
  const int3 Image_coord = { d_Image_min_indices.x + thread_id_x,
                             d_Image_min_indices.y + thread_id_y,
                             d_Image_min_indices.z + thread_id_z };

  // Get the index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x +
                         thread_id_y * d_Image_dim.x +
                         thread_id_x;
  const elemT val_center = image[inputIndex];

  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;

  
  elemT sum = 0.0;
  // Apply Gibbs prior over neighbourhood
  for (int dz = min_dz; dz <= max_dz; dz++)
      for (int dy = min_dy; dy <= max_dy; dy++)
          for (int dx = min_dx; dx <= max_dx; dx++)
            {
              //Linearized index for the neighbour voxel
              const int neighbourIndex = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x +
                                         (thread_id_y + dy) * d_Image_dim.x +
                                         (thread_id_x + dx);
              //Linearized index for the weight
              const int weightsIndex   = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x +
                                         (dy - d_weight_min_indices.y) * weights_size_x +
                                         (dx - d_weight_min_indices.x);
              //Neighbour voxel contribution
              const elemT val_neigh    = image[neighbourIndex];
              elemT current         = weights[weightsIndex]
                                      * cuda_potential_fun.derivative_20(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

              if (do_kappa)
                {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              sum += current;
            }
  Hessian_diag[inputIndex] = sum * 2.0 * penalisation_factor;

  // atomicAdd(global_sum, sum);
  // atomicAddDouble(global_sum, sum);
}

template <class elemT, typename CudaPotentialT>
__global__ static void
Hessian_Times_Input_Kernel(elemT* tmp_output,
                           const elemT* current_estimate,
                           const elemT* input,
                           const float* weights,
                           const elemT* kappa,
                           const bool do_kappa,
                           const float penalisation_factor,
                           const int3 Image_dim,
                           const int3 Image_max_indices,
                           const int3 Image_min_indices,
                           const int3 weight_max_indices,
                           const int3 weight_min_indices)
{
  // Get the voxel in x, y, z dimensions
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the voxel is within the image dimensions
  if (z >= Image_dim.z || y >= Image_dim.y || x >= Image_dim.x)
    return;

  // Get the index of the voxel
  const int inputIndex = z * Image_dim.y * Image_dim.x + y * Image_dim.x + x;

  // COMMENTED OUT FOR TESTING - Just set Hessian output to zero
  /*
  const elemT val_center = current_estimate[inputIndex];
  const elemT input_center = input[inputIndex];

  // Define Hessian times input variable
  double hessian_times_input = 0.0;

  // Calculate neighbourhood bounds
  const int min_dz = max(weight_min_z, -z);
  const int max_dz = min(weight_max_z, z_dim - 1 - z);
  const int min_dy = max(weight_min_y, -y);
  const int max_dy = min(weight_max_y, y_dim - 1 - y);
  const int min_dx = max(weight_min_x, -x);
  const int max_dx = min(weight_max_x, x_dim - 1 - x);

  // Apply Gibbs prior Hessian computation over neighbourhood
  for (int dz = min_dz; dz <= max_dz; dz++) {
    for (int dy = min_dy; dy <= max_dy; dy++) {
      for (int dx = min_dx; dx <= max_dx; dx++) {
              const int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
              const int weightsIndex = (dz - weight_min_z) * (weight_max_y - weight_min_y + 1) * (weight_max_x - weight_min_x + 1)
                                     + (dy - weight_min_y) * (weight_max_x - weight_min_x + 1)
                                     + (dx - weight_min_x);

              const elemT val_neigh = current_estimate[neighbourIndex];
              const elemT input_neigh = input[neighbourIndex];

        double current = weights[weightsIndex] * potential_fun.hessian_times_input(val_center, val_neigh, input_center,
  input_neigh, make_coordinate(z, y, x));

        if (do_kappa) {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              hessian_times_input += current;
            }
        }
    }
  */

  tmp_output[inputIndex] = static_cast<elemT>(0.0); // Just return zero for now
}

// Template method implementations

template <typename elemT, typename CudaPotentialT>
CudaGibbsPrior<elemT, CudaPotentialT>::CudaGibbsPrior()
    : base_type(), // Call parent constructor
      d_weights_data(nullptr),
      d_kappa_data(nullptr) {}

template <typename elemT, typename CudaPotentialT>
CudaGibbsPrior<elemT, CudaPotentialT>::CudaGibbsPrior(const bool only_2D, float penalization_factor)
    : base_type(only_2D, penalization_factor), // Call parent constructor
      d_weights_data(nullptr),
      d_kappa_data(nullptr) {}

template <typename elemT, typename CudaPotentialT>
CudaGibbsPrior<elemT, CudaPotentialT>::~CudaGibbsPrior()
{
  if (d_weights_data)
    cudaFree(d_weights_data);
  if (d_kappa_data)
    cudaFree(d_kappa_data);
}

template <typename elemT, typename CudaPotentialT>
double
CudaGibbsPrior<elemT, CudaPotentialT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate)
{

  if (this->penalisation_factor == 0)
    return 0.;
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("CudaGibbsPrior: set_up has not been called");

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  int threads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t shared_mem_bytes = threads_per_block * sizeof(elemT);

  elemT* d_image_data;
  double* d_prior_value;

  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_prior_value, sizeof(double));

  cudaMemset(d_prior_value, 0, sizeof(double));

  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_value kernel execution: ") + err);
    }

  array_to_device(d_image_data, current_image_estimate);
  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(d_kappa_data)))
    error("CudaGibbsPrior internal error: inconsistent CPU and device kappa");


  Value_Kernel<elemT, CudaPotentialT>
      <<<cuda_grid_dim, cuda_block_dim, shared_mem_bytes>>>(d_prior_value,
                                                            d_image_data,
                                                            d_weights_data,
                                                            do_kappa ? d_kappa_data : nullptr,
                                                            do_kappa,
                                                            d_Image_dim,
                                                            d_Image_max_indices,
                                                            d_Image_min_indices,
                                                            d_weight_max_indices,
                                                            d_weight_min_indices,
                                                            potential);

  double prior_value;
  cudaMemcpy(&prior_value, d_prior_value, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_image_data);
  cudaFree(d_prior_value);

  return this->penalisation_factor * prior_value;
}

template <typename elemT, typename CudaPotentialT>
void
CudaGibbsPrior<elemT, CudaPotentialT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                                                        const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  assert(prior_gradient.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor == 0)
    {
      prior_gradient.fill(0);
      return;
    }

  this->check(current_image_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaGibbsPrior: set_up has not been called");
    }

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  int tile_size = (block_dim.x+2) * (block_dim.y+2) * (block_dim.z+2);
  size_t shared_mem_bytes = tile_size * sizeof(elemT);

  // elemT *d_image_data, *d_gradient_data;

  // //Allocate memory on the GPU
  // cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  // cudaMalloc(&d_gradient_data, prior_gradient.size_all() * sizeof(elemT));
  // {
  //   cudaError_t cuda_error = cudaGetLastError();
  //   if (cuda_error != cudaSuccess)
  //     {
  //       const char* err = cudaGetErrorString(cuda_error);
  //       error(std::string("CUDA failed to allocate memory in compute_gradient kernel execution: ") + err);
  //     }
  // }

  // //Copy data from host to device
  // array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(d_kappa_data)))
    error("CudaGibbsPrior internal error: inconsistent CPU and device kappa");

  Gradient_Kernel<elemT, CudaPotentialT><<<cuda_grid_dim, cuda_block_dim>>>(d_gradient_data,
                                                                              d_image_data,
                                                                              d_weights_data,
                                                                              do_kappa ? d_kappa_data : nullptr,
                                                                              do_kappa,
                                                                              this->penalisation_factor,
                                                                              d_Image_dim,
                                                                              d_Image_max_indices,
                                                                              d_Image_min_indices,
                                                                              d_weight_max_indices,
                                                                              d_weight_min_indices,
                                                                              potential);  
  // Gradient_exp_Kernel<elemT, CudaPotentialT><<<cuda_grid_dim, cuda_block_dim, shared_mem_bytes>>>(d_gradient_data,
  //                                                                             d_image_data,
  //                                                                             d_weights_data,
  //                                                                             do_kappa ? d_kappa_data : nullptr,
  //                                                                             do_kappa,
  //                                                                             this->penalisation_factor,
  //                                                                             d_Image_dim,
  //                                                                             d_Image_max_indices,
  //                                                                             d_Image_min_indices,
  //                                                                             d_weight_max_indices,
  //                                                                             d_weight_min_indices,
  //                                                                             potential);


  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_gradient_data);
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CUDA error in compute_gradient kernel execution: ") + err);
    }

  //array_to_host(prior_gradient, d_gradient_data);

  // Cleanup
  // cudaFree(d_image_data);
  // cudaFree(d_gradient_data);

  // // Optional: write gradient to file
  // static int counter = 0;
  // if (this->gradient_filename_prefix.size() > 0)
  //   write_to_file(this->gradient_filename_prefix + std::to_string(counter++), prior_gradient);
}

template <typename elemT, typename CudaPotentialT>
double
CudaGibbsPrior<elemT, CudaPotentialT>::compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                                                    const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  //Preliminary Checks
  this->check(current_image_estimate);  
  if (this->_already_set_up == false)
    error("CudaGibbsPrior: set_up has not been called");
  if (this->penalisation_factor == 0)
    return 0.;
  
  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  int threads_per_block = block_dim.x * block_dim.y * block_dim.z;
  size_t shared_mem_bytes = threads_per_block * sizeof(elemT);

  elemT* d_image_data;
  elemT* d_input_data;
  double* d_result;

  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_input_data, input.size_all() * sizeof(elemT));
  cudaMalloc(&d_result, sizeof(double));

  cudaMemset(d_result, 0, sizeof(double));

  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_value kernel execution: ") + err);
    }

  array_to_device(d_image_data, current_image_estimate);
  array_to_device(d_input_data, input);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaGibbsPrior internal error: inconsistent CPU and device kappa");


  Gradient_dot_input_Kernel<elemT, CudaPotentialT>
  <<<cuda_grid_dim, cuda_block_dim, shared_mem_bytes>>> (d_result,
                                                        d_input_data,
                                                        d_image_data,
                                                        d_weights_data,
                                                        do_kappa ? d_kappa_data : nullptr,
                                                        do_kappa,
                                                        d_Image_dim,
                                                        d_Image_max_indices,
                                                        d_Image_min_indices,
                                                        d_weight_max_indices,
                                                        d_weight_min_indices,
                                                        potential);

  double result;
  cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_image_data);
  cudaFree(d_input_data);
  cudaFree(d_result);

  return this->penalisation_factor * result;
}

template <typename elemT, typename CudaPotentialT>
void
CudaGibbsPrior<elemT, CudaPotentialT>::compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diag,
                                                                const DiscretisedDensity<3, elemT>& current_image_estimate) const
{
  assert(Hessian_diag.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor == 0)
    {
      Hessian_diag.fill(0);
      return;
    }
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
      error("CudaGibbsPrior: set_up has not been called");
    

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);
  elemT *d_image_data, *d_hessian_data;

  // Allocate memory on the GPU
  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_hessian_data, Hessian_diag.size_all() * sizeof(elemT));
  {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
      {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_hessian_diagonal kernel execution: ") + err);
      }
  }

  // Copy data from host to device
  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaGibbsPrior internal error: inconsistent CPU and device kappa");

  Hessian_diagonal_Kernel<elemT, CudaPotentialT><<<cuda_grid_dim, cuda_block_dim>>>(d_hessian_data,
                                                                              d_image_data,
                                                                              this->d_weights_data,
                                                                              do_kappa ? this->d_kappa_data : nullptr,
                                                                              do_kappa,
                                                                              this->penalisation_factor,
                                                                              d_Image_dim,
                                                                              d_Image_max_indices,
                                                                              d_Image_min_indices,
                                                                              d_weight_max_indices,
                                                                              d_weight_min_indices,
                                                                              potential);

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_hessian_data);
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CUDA error in compute_hessian_diagonal kernel execution: ") + err);
    }

  array_to_host(Hessian_diag, d_hessian_data);

  // Cleanup
  cudaFree(d_image_data);
  cudaFree(d_hessian_data);
}

template <typename elemT, typename CudaPotentialT>
void
CudaGibbsPrior<elemT, CudaPotentialT>::accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                                                        const DiscretisedDensity<3, elemT>& current_estimate,
                                                                        const DiscretisedDensity<3, elemT>& input) const
{
  assert(output.has_same_characteristics(current_estimate));
  assert(input.has_same_characteristics(current_estimate));

  if (this->penalisation_factor == 0)
    {
      return;
    }

  this->check(current_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaGibbsPrior: set_up has not been called");
    }

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  elemT *d_current_data, *d_input_data, *d_output_data;

  // Allocate memory on the GPU
  cudaMalloc(&d_current_data, current_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_input_data, input.size_all() * sizeof(elemT));
  cudaMalloc(&d_output_data, output.size_all() * sizeof(elemT));
  {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
      {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in accumulate_Hessian_times_input kernel execution: ") + err);
      }
  }

  // Copy data from host to device
  array_to_device(d_current_data, current_estimate);
  array_to_device(d_input_data, input);
  array_to_device(d_output_data, output); // Copy existing output for accumulation

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaGibbsPrior internal error: inconsistent CPU and device kappa");

  Hessian_Times_Input_Kernel<elemT, CudaPotentialT><<<cuda_grid_dim, cuda_block_dim>>>(d_output_data,
                                                                                         d_current_data,
                                                                                         d_input_data,
                                                                                         this->d_weights_data,
                                                                                         do_kappa ? this->d_kappa_data : nullptr,
                                                                                         do_kappa,
                                                                                         this->penalisation_factor,
                                                                                         d_Image_dim,
                                                                                         d_Image_max_indices,
                                                                                         d_Image_min_indices,
                                                                                         d_weight_max_indices,
                                                                                         d_weight_min_indices);

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_current_data);
      cudaFree(d_input_data);
      cudaFree(d_output_data);
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CUDA error in accumulate_Hessian_times_input kernel execution: ") + err);
    }

  array_to_host(output, d_output_data);

  // Cleanup
  cudaFree(d_current_data);
  cudaFree(d_input_data);
  cudaFree(d_output_data);
}

template <typename elemT, typename CudaPotentialT>
Succeeded
CudaGibbsPrior<elemT, CudaPotentialT>::set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr)
{

  if (base_type::set_up(target_sptr) == Succeeded::no)
    return Succeeded::no;
    this->_already_set_up = false;

  
  // Fill CUDA int3 objects (This is needed because CartesianCoordinate3D cannot be used on GPU for now)
  d_Image_dim = make_int3(this->Image_dim.x(), this->Image_dim.y(), this->Image_dim.z());
  d_Image_max_indices = make_int3(this->Image_max_indices.x(), this->Image_max_indices.y(), this->Image_max_indices.z());
  d_Image_min_indices = make_int3(this->Image_min_indices.x(), this->Image_min_indices.y(), this->Image_min_indices.z());
  d_weight_max_indices = make_int3(this->weight_max_indices.x(), this->weight_max_indices.y(), this->weight_max_indices.z());
  d_weight_min_indices = make_int3(this->weight_min_indices.x(), this->weight_min_indices.y(), this->weight_min_indices.z());
    
  // Set the thread block and grid dimensions
  block_dim.x = 16;
  block_dim.y = 16;
  block_dim.z = 2;

  grid_dim.x = (d_Image_dim.x + block_dim.x - 1) / block_dim.x;
  grid_dim.y = (d_Image_dim.y + block_dim.y - 1) / block_dim.y;
  grid_dim.z = (d_Image_dim.z + block_dim.z - 1) / block_dim.z;

  // Copy CPU weights to GPU (weights should already be set up by parent)
  if (this->weights.get_length() > 0)
    {
      if (d_weights_data)
        cudaFree(d_weights_data);
      cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float));
      array_to_device(d_weights_data, this->weights);
    }
  
  // Copy CPU kappa to GPU
  // TODO: check this block of code
  if (d_kappa_data)
    cudaFree(d_kappa_data);
  auto kappa_ptr = this->get_kappa_sptr();
  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa)
    {
      if (!kappa_ptr->has_same_characteristics(*target_sptr))
        error("CudaGibbsPrior: kappa image does not have the same index range as the reconstructed image");
      cudaMalloc(&d_kappa_data, kappa_ptr->size_all() * sizeof(elemT));
      array_to_device(d_kappa_data, *kappa_ptr);
    }
  
  this->_already_set_up = true;

    

  //Allocate memory on the GPU
  auto& target_cast = dynamic_cast<const VoxelsOnCartesianGrid<elemT>&>(*target_sptr);
  cudaMalloc(&d_image_data, target_cast.size_all() * sizeof(elemT));
  cudaMalloc(&d_gradient_data, target_cast.size_all() * sizeof(elemT));
  {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
      {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_gradient kernel execution: ") + err);
      }
  }

  //Copy data from host to device
  array_to_device(d_image_data, target_cast);

  return Succeeded::yes;
}

template <typename elemT, typename CudaPotentialT>
void
CudaGibbsPrior<elemT, CudaPotentialT>::set_weights(const Array<3, float>& w)
{
  base_type::set_weights(w);
  if (d_weights_data)
    cudaFree(d_weights_data);
  cudaMalloc(&d_weights_data, w.size_all() * sizeof(float));
  array_to_device(d_weights_data, w);
} 

template <typename elemT, typename CudaPotentialT>
void
CudaGibbsPrior<elemT, CudaPotentialT>::set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>& k)
{
  base_type::set_kappa_sptr(k);
  if (d_kappa_data)
    cudaFree(d_kappa_data);
  if (!is_null_ptr(k))
    {
      cudaMalloc(&d_kappa_data, k->size_all() * sizeof(elemT));
      array_to_device(d_kappa_data, *k);
    }
  else
    d_kappa_data = nullptr;
}

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_CUDA_CudaGibbsPrior_CUH__
