/*
    Copyright (C) 2020, 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief implementation of the stir::CudaRelativeDifferencePrior class

  \author Imraj Singh
  \author Kris Thielemans
  \author Robert Twyman
*/

#include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/cuda_utilities.h"
#include <cuda_runtime.h>
#include <numeric>

START_NAMESPACE_STIR

template <class elemT>
__global__ static void
computeCudaRelativeDifferencePriorGradientKernel(elemT* tmp_grad,
                                                 const elemT* image,
                                                 const float* weights,
                                                 const elemT* kappa,
                                                 const bool do_kappa,
                                                 const float gamma,
                                                 const float epsilon,
                                                 const float penalisation_factor,
                                                 const int z_dim,
                                                 const int y_dim,
                                                 const int x_dim)
{
  // Get the voxel in x, y, z dimensions
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the voxel is within the image dimensions
  if (z >= z_dim || y >= y_dim || x >= x_dim)
    return;

  // Get the index of the voxel
  const int inputIndex = z * y_dim * x_dim + y * x_dim + x;

  // Define a single voxel gradient variable
  double voxel_gradient = 0.0;

  // Define the neighbourhood
  int min_dz = -1;
  int max_dz = 1;
  int min_dy = -1;
  int max_dy = 1;
  int min_dx = -1;
  int max_dx = 1;

  // Check if the neighbourhood is at the boundary
  if (z == 0)
    min_dz = 0;
  if (z == z_dim - 1)
    max_dz = 0;
  if (y == 0)
    min_dy = 0;
  if (y == y_dim - 1)
    max_dy = 0;
  if (x == 0)
    min_dx = 0;
  if (x == x_dim - 1)
    max_dx = 0;

  // Apply RDP with hard coded 3x3x3 neighbourhood
  for (int dz = min_dz; dz <= max_dz; dz++)
    {
      for (int dy = min_dy; dy <= max_dy; dy++)
        {
          for (int dx = min_dx; dx <= max_dx; dx++)
            {
              const int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
              const int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
              const elemT diff = (image[inputIndex] - image[neighbourIndex]);
              const elemT diff_abs = std::abs(diff);
              const elemT add = (image[inputIndex] + image[neighbourIndex]);
              const elemT add_3 = (image[inputIndex] + 3 * image[neighbourIndex]);
              double current = weights[weightsIndex] * (diff * (gamma * diff_abs + add_3 + 2 * epsilon))
                               / ((add + gamma * diff_abs + epsilon) * (add + gamma * diff_abs + epsilon));
              if (do_kappa)
                {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              voxel_gradient += current;
            }
        }
    }
  tmp_grad[inputIndex] = static_cast<elemT>(penalisation_factor * voxel_gradient);
}

// typedef for the value (array) in the CUDA kernel
// We need double for numerical accuracy accordingy to test_priors.cxx
typedef double value_type;

template <class elemT>
__global__ static void
computeCudaRelativeDifferencePriorValueKernel(value_type* tmp_value,
                                              const elemT* image,
                                              const float* weights,
                                              const elemT* kappa,
                                              const bool do_kappa,
                                              const float gamma,
                                              const float epsilon,
                                              const float penalisation_factor,
                                              const int z_dim,
                                              const int y_dim,
                                              const int x_dim)
{
  // Get the voxel in x, y, z dimensions
  const int z = blockIdx.z * blockDim.z + threadIdx.z;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the voxel is within the image dimensions
  if (z >= z_dim || y >= y_dim || x >= x_dim)
    return;

  // Get the index of the voxel
  const int inputIndex = z * y_dim * x_dim + y * x_dim + x;

  // Define the sum variable
  double sum = 0.0f;

  // Define the neighbourhood
  int min_dz = -1;
  int max_dz = 1;
  int min_dy = -1;
  int max_dy = 1;
  int min_dx = -1;
  int max_dx = 1;

  // Check if the neighbourhood is at the boundary
  if (z == 0)
    min_dz = 0;
  if (z == z_dim - 1)
    max_dz = 0;
  if (y == 0)
    min_dy = 0;
  if (y == y_dim - 1)
    max_dy = 0;
  if (x == 0)
    min_dx = 0;
  if (x == x_dim - 1)
    max_dx = 0;

  // Apply RDP with hard coded 3x3x3 neighbourhood
  for (int dz = min_dz; dz <= max_dz; dz++)
    {
      for (int dy = min_dy; dy <= max_dy; dy++)
        {
          for (int dx = min_dx; dx <= max_dx; dx++)
            {
              const int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
              const int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);

              const elemT diff = (image[inputIndex] - image[neighbourIndex]);
              const elemT add = (image[inputIndex] + image[neighbourIndex]);
              double current = (weights[weightsIndex] * 0.5 * diff * diff) / (add + gamma * std::abs(diff) + epsilon);
              if (do_kappa)
                {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              sum += current;
            }
        }
    }
  tmp_value[inputIndex] = static_cast<value_type>(penalisation_factor * sum);
}

static void
compute_weights(Array<3, float>& weights, const CartesianCoordinate3D<float>& grid_spacing, const bool only_2D)
{
  int min_dz, max_dz;
  if (only_2D)
    {
      min_dz = max_dz = 0;
    }
  else
    {
      min_dz = -1;
      max_dz = 1;
    }
  weights = Array<3, float>(IndexRange3D(min_dz, max_dz, -1, 1, -1, 1));
  for (int z = min_dz; z <= max_dz; ++z)
    for (int y = -1; y <= 1; ++y)
      for (int x = -1; x <= 1; ++x)
        {
          if (z == 0 && y == 0 && x == 0)
            weights[0][0][0] = 0;
          else
            {
              weights[z][y][x]
                  = grid_spacing.x()
                    / sqrt(square(x * grid_spacing.x()) + square(y * grid_spacing.y()) + square(z * grid_spacing.z()));
            }
        }
}

template <typename elemT>
void
CudaRelativeDifferencePrior<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
CudaRelativeDifferencePrior<elemT>::~CudaRelativeDifferencePrior()
{
  if (this->d_weights_data)
    cudaFree(this->d_weights_data);
  if (this->d_kappa_data)
    cudaFree(this->d_kappa_data);
}

template <typename elemT>
void
CudaRelativeDifferencePrior<elemT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                                                     const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  assert(prior_gradient.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor == 0)
    {
      prior_gradient.fill(0);
      return;
    }

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  this->check(current_image_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaRelativeDifferencePrior: set_up has not been called");
    }

  elemT *d_image_data, *d_gradient_data;

  // Allocate memory on the GPU
  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_gradient_data, prior_gradient.size_all() * sizeof(elemT));
  {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
      {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_gradient kernel execution: ") + err);
      }
  }
  // Copy data from host to device
  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaRelativeDifferencePrior internal error: inconsistent CPU and device kappa");
  computeCudaRelativeDifferencePriorGradientKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_gradient_data,
                                                                                      d_image_data,
                                                                                      this->d_weights_data,
                                                                                      do_kappa ? this->d_kappa_data : nullptr,
                                                                                      do_kappa,
                                                                                      this->gamma,
                                                                                      this->epsilon,
                                                                                      this->penalisation_factor,
                                                                                      this->z_dim,
                                                                                      this->y_dim,
                                                                                      this->x_dim);

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_gradient_data);
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CUDA error in compute_gradient kernel execution: ") + err);
    }

  array_to_host(prior_gradient, d_gradient_data);
  // Cleanup
  cudaFree(d_image_data);
  cudaFree(d_gradient_data);
}

template <typename elemT>
double
CudaRelativeDifferencePrior<elemT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate)
{

  if (this->penalisation_factor == 0)
    {
      return 0.;
    }

  dim3 cuda_block_dim(block_dim.x, block_dim.y, block_dim.z);
  dim3 cuda_grid_dim(grid_dim.x, grid_dim.y, grid_dim.z);

  this->check(current_image_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaRelativeDifferencePrior: set_up has not been called");
    }

  // GPU memory pointers
  elemT* d_image_data;
  value_type* d_tmp_value;

  // Allocate memory on the GPU
  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(elemT));
  cudaMalloc(&d_tmp_value, current_image_estimate.size_all() * sizeof(value_type));
  {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
      {
        const char* err = cudaGetErrorString(cuda_error);
        error(std::string("CUDA failed to allocate memory in compute_value kernel execution: ") + err);
      }
  }

  // Copy data from host to device
  array_to_device(d_image_data, current_image_estimate);

  const bool do_kappa = !is_null_ptr(this->get_kappa_sptr());
  if (do_kappa != (!is_null_ptr(this->d_kappa_data)))
    error("CudaRelativeDifferencePrior internal error: inconsistent CPU and device kappa");
  // Launch the kernel
  computeCudaRelativeDifferencePriorValueKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_tmp_value,
                                                                                   d_image_data,
                                                                                   this->d_weights_data,
                                                                                   do_kappa ? this->d_kappa_data : nullptr,
                                                                                   do_kappa,
                                                                                   this->gamma,
                                                                                   this->epsilon,
                                                                                   this->penalisation_factor,
                                                                                   this->z_dim,
                                                                                   this->y_dim,
                                                                                   this->x_dim);

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_tmp_value);
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CUDA error in compute_value kernel execution: ") + err);
    }

  // Allocate host memory for the result and copy from device to host
  Array<3, value_type> tmp_value(current_image_estimate.get_index_range());
  array_to_host(tmp_value, d_tmp_value);

  // Compute the total value from tmp_value
  double totalValue = tmp_value.sum();

  // Cleanup
  cudaFree(d_image_data);
  cudaFree(d_tmp_value);
  return totalValue;
}

template <typename elemT>
Succeeded
CudaRelativeDifferencePrior<elemT>::set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr)
{
  if (RelativeDifferencePrior<elemT>::set_up(target_sptr) == Succeeded::no)
    {
      return Succeeded::no;
    }
  // Get the number of voxels in each dimension
  auto& target_cast = dynamic_cast<const VoxelsOnCartesianGrid<elemT>&>(*target_sptr);
  auto sizes = target_cast.get_lengths();
  this->z_dim = sizes[1]; // note: 1-offset in BasicCoordinate sadly
  this->y_dim = sizes[2];
  this->x_dim = sizes[3];

  // Set the thread block and grid dimensions using std::tuple
  this->block_dim.x = 8;
  this->block_dim.y = 8;
  this->block_dim.z = 8;

  this->grid_dim.x = (this->x_dim + this->block_dim.x - 1) / this->block_dim.x;
  this->grid_dim.y = (this->y_dim + this->block_dim.y - 1) / this->block_dim.y;
  this->grid_dim.z = (this->z_dim + this->block_dim.z - 1) / this->block_dim.z;
  // this->block_dim = dim3(8, 8, 8);
  // this->grid_dim = dim3((this->x_dim + this->block_dim.x - 1) / this->block_dim.x, (this->y_dim + this->block_dim.y - 1) /
  // this->block_dim.y, (this->z_dim + this->block_dim.z - 1) / this->block_dim.z);
  //  Check if z_dim is 1 or only 2D is true and return an error if it is
  if (this->z_dim == 1 || this->only_2D)
    {
      error("CudaRelativeDifferencePriorClass: This prior requires a 3D image and only works for a 3x3x3 neighbourhood");
      return Succeeded::no;
    }
  if (this->weights.get_length() == 0)
    {
      compute_weights(this->weights, target_cast.get_grid_spacing(), this->only_2D);
      if (this->d_weights_data)
        cudaFree(this->d_weights_data);
      cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float));
      array_to_device(this->d_weights_data, this->weights);
    }
  // check 3x3x3 (kernel limitation)
  {
    BasicCoordinate<3, int> min_ind, max_ind;
    if (!this->weights.get_regular_range(min_ind, max_ind) || (min_ind != make_coordinate(-1, -1, -1))
        || (max_ind != make_coordinate(1, 1, 1)))
      error("CudaRelativeDifferencePrior: Currently only support 3x3x3 weights. Sorry.");
  }

  {
    if (this->d_kappa_data)
      cudaFree(this->d_kappa_data);
    auto kappa_ptr = this->get_kappa_sptr();
    const bool do_kappa = !is_null_ptr(kappa_ptr);
    if (do_kappa)
      {
        if (!kappa_ptr->has_same_characteristics(*target_sptr))
          error("RelativeDifferencePrior: kappa image does not have the same index range as the reconstructed image");
        cudaMalloc(&this->d_kappa_data, kappa_ptr->size_all() * sizeof(elemT));
        array_to_device(this->d_kappa_data, *kappa_ptr);
      }
  }
  this->_already_set_up = true;
  return Succeeded::yes;
}

template class CudaRelativeDifferencePrior<float>;

END_NAMESPACE_STIR
