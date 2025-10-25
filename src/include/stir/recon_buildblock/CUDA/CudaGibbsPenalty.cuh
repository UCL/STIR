//
//
/*
    Copyright (C) 2025, University of Milano-Bicocca
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Implementation of the stir::CudaGibbsPenalty class

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_CUDA_CudaGibbsPenalty_CUH__
#define __stir_recon_buildblock_CUDA_CudaGibbsPenalty_CUH__

#include "stir/recon_buildblock/CUDA/CudaGibbsPenalty.h"

#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/Array.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/warning.h"
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>

START_NAMESPACE_STIR

// CUDA kernels for CudaGibbsPenalty

//! CUDA kernel for Gibbs penalty value calculation using the potential function value and block reduction
/*!
 * Each thread processes one voxel, computes local sum, then performs parallel reduction in shared memory.
 * Result from each block is then atomically added to output. Requires CUDA compute capability ≥6.0 for double atomics.
 */
template <class elemT, typename PotentialT>
__global__ static void
CudaGibbsPenalty_value_kernel(double* output,
                            const elemT* __restrict__ current_image,
                            const float* __restrict__ weights,
                            const elemT* __restrict__ kappa,
                            const bool do_kappa,
                            const int3 d_Image_dim,
                            const int3 d_Image_max_indices,
                            const int3 d_Image_min_indices,
                            const int3 d_weight_max_indices,
                            const int3 d_weight_min_indices,
                            const PotentialT potential)
{
  extern __shared__ elemT block_sum_shared[];

  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // local linearized index inside the block
  int thread_in_block = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  // Initialize shared memory for block sum
  block_sum_shared[thread_in_block] = elemT(0);

  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;
  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };

  // Global linearized index of the voxel (memory access is still 0-based)
  const int centerIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + thread_id_y * d_Image_dim.x + thread_id_x;

  const elemT val_center = current_image[centerIndex];

  elemT sum = 0.0;
  // Boundary conditions for the neighbourhood
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
          const int neighbourIndex
              = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x + (thread_id_y + dy) * d_Image_dim.x + (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x
                                   + (dy - d_weight_min_indices.y) * weights_size_x + (dx - d_weight_min_indices.x);

          const elemT val_neigh = current_image[neighbourIndex];
          elemT current
              = weights[weightsIndex] * potential.value(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            current *= kappa[centerIndex] * kappa[neighbourIndex];

          sum += current;
        }
  // Each thread puts its partial sum in shared memory
  block_sum_shared[thread_in_block] = sum;
  // Local sync on the threads in the block
  __syncthreads();

  // block pair-wise reduction in shared memory
  int block_threads = blockDim.x * blockDim.y * blockDim.z;
  blockReduction(block_sum_shared, thread_in_block, block_threads);

  // One thread per block performs final atomic add in double
  if (thread_in_block == 0)
    atomicAddGeneric(output, block_sum_shared[0]);
}

//! CUDA kernel for Gibbs penalty gradient calculation using the potential function first derivative.
/*!
 * Each thread processes one voxel, and computes one element of the gradient.
 */
template <class elemT, typename PotentialT>
__global__ static void
CudaGibbsPenalty_gradient_kernel(elemT* gradient,
                               const elemT* __restrict__ current_image,
                               const float* __restrict__ weights,
                               const elemT* __restrict__ kappa,
                               const bool do_kappa,
                               const float penalisation_factor,
                               const int3 d_Image_dim,
                               const int3 d_Image_max_indices,
                               const int3 d_Image_min_indices,
                               const int3 d_weight_max_indices,
                               const int3 d_weight_min_indices,
                               const PotentialT potential)
{
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;
  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };
  // Global linearized index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + thread_id_y * d_Image_dim.x + thread_id_x;

  const elemT val_center = current_image[inputIndex];

  // Boundary conditions for the neighbourhood
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
          // Linearized index for the neighbour voxel
          const int neighbourIndex
              = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x + (thread_id_y + dy) * d_Image_dim.x + (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x
                                   + (dy - d_weight_min_indices.y) * weights_size_x + (dx - d_weight_min_indices.x);
          // Neighbour voxel contribution
          const elemT val_neigh = current_image[neighbourIndex];

          elemT current = weights[weightsIndex]
                          * potential.derivative_10(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            current *= kappa[inputIndex] * kappa[neighbourIndex];

          sum += current;
        }
  gradient[inputIndex] = sum * 2.0 * penalisation_factor;
}

//! CUDA kernel for Gibbs penalty, computes the dot product of the gradient with a given input image using the potential function
//! first derivative.
/*!
 * Each thread computes one element of the gradient, then we perform the dot product in shared memory.
 * Result from each block is then atomically added to output. Requires CUDA compute capability ≥6.0 for double atomics.
 */
template <class elemT, typename PotentialT>
__global__ static void
CudaGibbsPenalty_gradient_dot_input_kernel(double* output,
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
                                         const PotentialT potential)
{
  extern __shared__ elemT block_sum_shared[];

  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // local linearized index inside the block
  int thread_in_block = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

  block_sum_shared[thread_in_block] = elemT(0);

  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;

  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };

  // Global linearized index of the voxel (memory access is still 0-based)
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
          // Linearized index for the neighbour voxel
          const int neighbourIndex
              = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x + (thread_id_y + dy) * d_Image_dim.x + (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x
                                   + (dy - d_weight_min_indices.y) * weights_size_x + (dx - d_weight_min_indices.x);
          // Neighbour voxel contribution
          const elemT val_neigh = current_image[neighbourIndex];

          elemT current = weights[weightsIndex]
                          * potential.derivative_10(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            current *= kappa[inputIndex] * kappa[neighbourIndex];

          sum += current;
        }
  // Each thread puts its partial sum in shared memory
  block_sum_shared[thread_in_block] = 2 * sum * input[inputIndex];
  // Local sync on the threads in the block
  __syncthreads();

  // block pair-wise reduction in shared memory
  int block_threads = blockDim.x * blockDim.y * blockDim.z;
  blockReduction(block_sum_shared, thread_in_block, block_threads);

  if (thread_in_block == 0)
    atomicAddGeneric(output, block_sum_shared[0]);
}

//! CUDA kernel for Gibbs penalty, computes the Hessian diagonal using the potential function second derivative.
/*!
 * Each thread computes one element of the Hessian diagonal.
 */
template <class elemT, typename PotentialT>
__global__ static void
CudaGibbsPenalty_Hessian_diagonal_kernel(elemT* Hessian_diag,
                                       const elemT* __restrict__ current_image,
                                       const float* __restrict__ weights,
                                       const elemT* __restrict__ kappa,
                                       const bool do_kappa,
                                       const float penalisation_factor,
                                       const int3 d_Image_dim,
                                       const int3 d_Image_max_indices,
                                       const int3 d_Image_min_indices,
                                       const int3 d_weight_max_indices,
                                       const int3 d_weight_min_indices,
                                       const PotentialT potential)
{
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;

  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };

  // Global linearized index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + thread_id_y * d_Image_dim.x + thread_id_x;
  const elemT val_center = current_image[inputIndex];

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
          // Linearized index for the neighbour voxel
          const int neighbourIndex
              = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x + (thread_id_y + dy) * d_Image_dim.x + (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x
                                   + (dy - d_weight_min_indices.y) * weights_size_x + (dx - d_weight_min_indices.x);
          // Neighbour voxel contribution
          const elemT val_neigh = current_image[neighbourIndex];
          elemT current = weights[weightsIndex]
                          * potential.derivative_20(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x);

          if (do_kappa)
            {
              current *= kappa[inputIndex] * kappa[neighbourIndex];
            }
          sum += current;
        }
  Hessian_diag[inputIndex] = sum * 2.0 * penalisation_factor;
}

//! CUDA kernel for Gibbs penalty, computes the Hessian times a given input image using the potential function second derivatives.
/*!
 * The output is image shaped and each element is computed by a given thread.
 */
template <class elemT, typename PotentialT>
__global__ static void
CudaGibbsPenalty_Hessian_Times_Input_kernel(elemT* output,
                                          const elemT* __restrict__ current_image,
                                          const elemT* __restrict__ input,
                                          const float* __restrict__ weights,
                                          const elemT* __restrict__ kappa,
                                          const bool do_kappa,
                                          const float penalisation_factor,
                                          const int3 d_Image_dim,
                                          const int3 d_Image_max_indices,
                                          const int3 d_Image_min_indices,
                                          const int3 d_weight_max_indices,
                                          const int3 d_weight_min_indices,
                                          const PotentialT potential)
{
  const int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread_id_z = blockIdx.z * blockDim.z + threadIdx.z;

  // Check if the thread is within the image dimensions
  if (thread_id_z >= d_Image_dim.z || thread_id_y >= d_Image_dim.y || thread_id_x >= d_Image_dim.x)
    return;
  // Convert thread indices to actual image coordinates
  const int3 Image_coord
      = { d_Image_min_indices.x + thread_id_x, d_Image_min_indices.y + thread_id_y, d_Image_min_indices.z + thread_id_z };

  // Global linearized index of the voxel (memory access is still 0-based)
  const int inputIndex = thread_id_z * d_Image_dim.y * d_Image_dim.x + thread_id_y * d_Image_dim.x + thread_id_x;

  const elemT val_center = current_image[inputIndex];
  const elemT input_center = input[inputIndex];
  // Boundary conditions for the neighbourhood
  const int min_dz = max(d_weight_min_indices.z, d_Image_min_indices.z - Image_coord.z);
  const int max_dz = min(d_weight_max_indices.z, d_Image_max_indices.z - Image_coord.z);
  const int min_dy = max(d_weight_min_indices.y, d_Image_min_indices.y - Image_coord.y);
  const int max_dy = min(d_weight_max_indices.y, d_Image_max_indices.y - Image_coord.y);
  const int min_dx = max(d_weight_min_indices.x, d_Image_min_indices.x - Image_coord.x);
  const int max_dx = min(d_weight_max_indices.x, d_Image_max_indices.x - Image_coord.x);
  const int weights_size_y = d_weight_max_indices.y - d_weight_min_indices.y + 1;
  const int weights_size_x = d_weight_max_indices.x - d_weight_min_indices.x + 1;

  // Define variable to sum over the neighbourhood
  elemT result = 0;
  for (int dz = min_dz; dz <= max_dz; dz++)
    for (int dy = min_dy; dy <= max_dy; dy++)
      for (int dx = min_dx; dx <= max_dx; dx++)
        {
          // Linearized index for the neighbour voxel
          const int neighbourIndex
              = (thread_id_z + dz) * d_Image_dim.y * d_Image_dim.x + (thread_id_y + dy) * d_Image_dim.x + (thread_id_x + dx);
          // Linearized index for the weight
          const int weightsIndex = (dz - d_weight_min_indices.z) * weights_size_y * weights_size_x
                                   + (dy - d_weight_min_indices.y) * weights_size_x + (dx - d_weight_min_indices.x);
          // Neighbour voxel contribution
          elemT current = weights[weightsIndex];
          const elemT val_neigh = current_image[neighbourIndex];
          const elemT input_neigh = input[neighbourIndex];

          if ((dz == 0) && (dy == 0) && (dx == 0))
            current *= potential.derivative_20(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x) * input_center;

          else
            current
                *= potential.derivative_20(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x) * input_center
                   + potential.derivative_11(val_center, val_neigh, Image_coord.z, Image_coord.y, Image_coord.x) * input_neigh;

          if (do_kappa)
            current *= kappa[inputIndex] * kappa[neighbourIndex];

          result += current;
        }
  output[inputIndex] += 2.0 * result * penalisation_factor;
}

template <typename elemT, typename PotentialT>
CudaGibbsPenalty<elemT, PotentialT>::CudaGibbsPenalty()
    : base_type(), // Call parent constructor
      d_weights_data(nullptr),
      d_kappa_data(nullptr)
{}

template <typename elemT, typename PotentialT>
CudaGibbsPenalty<elemT, PotentialT>::CudaGibbsPenalty(const bool only_2D, float penalization_factor)
    : base_type(only_2D, penalization_factor), // Call parent constructor
      d_weights_data(nullptr),
      d_kappa_data(nullptr)
{}

template <typename elemT, typename PotentialT>
CudaGibbsPenalty<elemT, PotentialT>::~CudaGibbsPenalty()
{
  if (d_weights_data)
    cudaFree(d_weights_data);
  if (d_kappa_data)
    cudaFree(d_kappa_data);
  if (d_image_data)
    cudaFree(d_image_data);
  if (d_scalar)
    cudaFree(d_scalar);
  if (d_output_data)
    cudaFree(d_output_data);
  if (d_input_data)
    cudaFree(d_input_data);
}

template <typename elemT, typename PotentialT>
double
CudaGibbsPenalty<elemT, PotentialT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate)
{

  if (this->penalisation_factor == 0)
    return 0.;
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("CudaGibbsPenalty: set_up has not been called");

  cudaMemset(d_scalar, 0, sizeof(double));
  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(d_kappa_data)))
    error("CudaGibbsPenalty internal error: inconsistent CPU and device kappa");

  CudaGibbsPenalty_value_kernel<elemT, PotentialT><<<grid_dim, block_dim, shared_mem_bytes>>>(d_scalar,
                                                                                            d_image_data,
                                                                                            d_weights_data,
                                                                                            do_kappa ? d_kappa_data : nullptr,
                                                                                            do_kappa,
                                                                                            d_image_dim,
                                                                                            d_image_max_indices,
                                                                                            d_image_min_indices,
                                                                                            d_weight_max_indices,
                                                                                            d_weight_min_indices,
                                                                                            this->potential);

  checkCudaError("compute_value kernel");

  double prior_value;
  cudaMemcpy(&prior_value, d_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  return this->penalisation_factor * prior_value;
}

template <typename elemT, typename PotentialT>
void
CudaGibbsPenalty<elemT, PotentialT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
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
    error("CudaGibbsPenalty: set_up has not been called");

  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(d_kappa_data)))
    error("CudaGibbsPenalty internal error: inconsistent CPU and device kappa");

  CudaGibbsPenalty_gradient_kernel<elemT, PotentialT><<<grid_dim, block_dim>>>(d_output_data,
                                                                             d_image_data,
                                                                             d_weights_data,
                                                                             do_kappa ? d_kappa_data : nullptr,
                                                                             do_kappa,
                                                                             this->penalisation_factor,
                                                                             d_image_dim,
                                                                             d_image_max_indices,
                                                                             d_image_min_indices,
                                                                             d_weight_max_indices,
                                                                             d_weight_min_indices,
                                                                             this->potential);

  checkCudaError("compute_gradient kernel");
  cudaDeviceSynchronize();
  array_to_host(prior_gradient, d_output_data);

  // Optional: write gradient to file
  static int counter = 0;
  if (this->gradient_filename_prefix.size() > 0)
    write_to_file(this->gradient_filename_prefix + std::to_string(counter++), prior_gradient);
}

template <typename elemT, typename PotentialT>
double
CudaGibbsPenalty<elemT, PotentialT>::compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                                                 const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  // Preliminary Checks
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("CudaGibbsPenalty: set_up has not been called");
  if (this->penalisation_factor == 0)
    return 0.;

  cudaMemset(d_scalar, 0, sizeof(double));

  array_to_device(d_image_data, current_image_estimate);
  array_to_device(d_input_data, input);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaGibbsPenalty internal error: inconsistent CPU and device kappa");

  CudaGibbsPenalty_gradient_dot_input_kernel<elemT, PotentialT>
      <<<grid_dim, block_dim, shared_mem_bytes>>>(d_scalar,
                                                  d_input_data,
                                                  d_image_data,
                                                  d_weights_data,
                                                  do_kappa ? d_kappa_data : nullptr,
                                                  do_kappa,
                                                  d_image_dim,
                                                  d_image_max_indices,
                                                  d_image_min_indices,
                                                  d_weight_max_indices,
                                                  d_weight_min_indices,
                                                  this->potential);

  checkCudaError("compute_gradient_times_input kernel");
  cudaDeviceSynchronize();
  double result;
  cudaMemcpy(&result, d_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  return this->penalisation_factor * result;
}

template <typename elemT, typename PotentialT>
void
CudaGibbsPenalty<elemT, PotentialT>::compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diag,
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
    error("CudaGibbsPenalty: set_up has not been called");

  // Copy data from host to device
  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaGibbsPenalty internal error: inconsistent CPU and device kappa");

  CudaGibbsPenalty_Hessian_diagonal_kernel<elemT, PotentialT><<<grid_dim, block_dim>>>(d_output_data,
                                                                                     d_image_data,
                                                                                     d_weights_data,
                                                                                     do_kappa ? d_kappa_data : nullptr,
                                                                                     do_kappa,
                                                                                     this->penalisation_factor,
                                                                                     d_image_dim,
                                                                                     d_image_max_indices,
                                                                                     d_image_min_indices,
                                                                                     d_weight_max_indices,
                                                                                     d_weight_min_indices,
                                                                                     this->potential);

  checkCudaError("compute_hessian_diagonal kernel");

  array_to_host(Hessian_diag, d_output_data);
}

template <typename elemT, typename PotentialT>
void
CudaGibbsPenalty<elemT, PotentialT>::accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                                                  const DiscretisedDensity<3, elemT>& current_image_estimate,
                                                                  const DiscretisedDensity<3, elemT>& input) const
{
  assert(output.has_same_characteristics(current_image_estimate));
  assert(input.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor == 0)
    return;

  this->check(current_image_estimate);

  if (this->_already_set_up == false)
    error("CudaGibbsPenalty: set_up has not been called");

  // Copy data from host to device
  array_to_device(d_image_data, current_image_estimate);
  array_to_device(d_input_data, input);
  array_to_device(d_output_data, output);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(d_kappa_data)))
    error("CudaGibbsPenalty internal error: inconsistent CPU and device kappa");

  CudaGibbsPenalty_Hessian_Times_Input_kernel<elemT, PotentialT><<<grid_dim, block_dim>>>(d_output_data,
                                                                                        d_image_data,
                                                                                        d_input_data,
                                                                                        d_weights_data,
                                                                                        do_kappa ? d_kappa_data : nullptr,
                                                                                        do_kappa,
                                                                                        this->penalisation_factor,
                                                                                        d_image_dim,
                                                                                        d_image_max_indices,
                                                                                        d_image_min_indices,
                                                                                        d_weight_max_indices,
                                                                                        d_weight_min_indices,
                                                                                        this->potential);

  checkCudaError("accumulate_Hessian_times_input kernel");

  array_to_host(output, d_output_data);
}

template <typename elemT, typename PotentialT>
Succeeded
CudaGibbsPenalty<elemT, PotentialT>::set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr)
{

  if (base_type::set_up(target_sptr) == Succeeded::no)
    return Succeeded::no;
  this->_already_set_up = false;

  // Fill CUDA int3 objects (This is needed because CartesianCoordinate3D cannot be used on GPU)
  // This is not very elegant but I don't see a better solution for now
  d_image_dim = make_int3(this->image_dim.x(), this->image_dim.y(), this->image_dim.z());
  d_image_max_indices = make_int3(this->image_max_indices.x(), this->image_max_indices.y(), this->image_max_indices.z());
  d_image_min_indices = make_int3(this->image_min_indices.x(), this->image_min_indices.y(), this->image_min_indices.z());
  d_weight_max_indices = make_int3(this->weights.get_max_index(), this->weights[0].get_max_index(), this->weights[0][0].get_max_index());
  d_weight_min_indices = make_int3(this->weights.get_min_index(), this->weights[0].get_min_index(), this->weights[0][0].get_min_index());

  // Set the thread block and grid dimensions
  block_dim.x = 8;
  block_dim.y = 8;
  block_dim.z = 1;

  grid_dim.x = (d_image_dim.x + block_dim.x - 1) / block_dim.x;
  grid_dim.y = (d_image_dim.y + block_dim.y - 1) / block_dim.y;
  grid_dim.z = (d_image_dim.z + block_dim.z - 1) / block_dim.z;

  // Variable used for shared memory to compute block reduction sums
  threads_per_block = block_dim.x * block_dim.y * block_dim.z;
  shared_mem_bytes = threads_per_block * sizeof(elemT);

  // Pre-allocate GPU memory for current image estimate
  if (d_image_data)
    cudaFree(d_image_data);
  cudaMalloc(&d_image_data, target_sptr->size_all() * sizeof(elemT));
  checkCudaError("CudaGibbsPenalty: cudaMalloc for d_image_data");

  // Pre-allocate GPU memory for input data
  if (d_input_data)
    cudaFree(d_input_data);
  cudaMalloc(&d_input_data, target_sptr->size_all() * sizeof(elemT));
  checkCudaError("CudaGibbsPenalty: cudaMalloc for d_input_data");

  // Pre-allocate GPU memory for outputs
  if (d_scalar)
    cudaFree(d_scalar);
  cudaMalloc(&d_scalar, sizeof(double));
  checkCudaError("CudaGibbsPenalty: cudaMalloc for d_scalar");

  if (d_output_data)
    cudaFree(d_output_data);
  cudaMalloc(&d_output_data, target_sptr->size_all() * sizeof(elemT));
  checkCudaError("CudaGibbsPenalty: cudaMalloc for d_output_data");

  // Copy CPU weights to GPU (weights should already be set up by parent)
  if (this->weights.get_length() > 0)
    {
      if (d_weights_data)
        cudaFree(d_weights_data);
      cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float));
      array_to_device(d_weights_data, this->weights);
      checkCudaError("CudaGibbsPenalty: cudaMalloc for d_weights_data");
    }

  // Copy CPU kappa image to GPU
  if (d_kappa_data)
    cudaFree(d_kappa_data);
  auto kappa_ptr = this->get_kappa_sptr();
  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa)
    {
      if (!kappa_ptr->has_same_characteristics(*target_sptr))
        error("CudaGibbsPenalty: kappa image does not have the same index range as the reconstructed image");
      cudaMalloc(&d_kappa_data, kappa_ptr->size_all() * sizeof(elemT));
      array_to_device(d_kappa_data, *kappa_ptr);
    }

  this->_already_set_up = true;
  return Succeeded::yes;
}

template <typename elemT, typename PotentialT>
void
CudaGibbsPenalty<elemT, PotentialT>::set_weights(const Array<3, float>& w)
{
  base_type::set_weights(w);
  if (d_weights_data)
    cudaFree(d_weights_data);
  cudaMalloc(&d_weights_data, w.size_all() * sizeof(float));
  array_to_device(d_weights_data, w);
}

template <typename elemT, typename PotentialT>
void
CudaGibbsPenalty<elemT, PotentialT>::set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>& k)
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

#endif // __stir_recon_buildblock_CUDA_CudaGibbsPenalty_CUH__
