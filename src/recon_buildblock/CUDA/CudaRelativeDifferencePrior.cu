#include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include <cuda_runtime.h>
#include <numeric>

extern "C" __global__ void
computeCudaRelativeDifferencePriorGradientKernel(float* tmp_grad,
                                                 const float* image,
                                                 const float* weights,
                                                 const float* kappa,
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
  double voxel_gradient = 0.0f;

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
              const float diff = (image[inputIndex] - image[neighbourIndex]);
              const float diff_abs = abs(diff);
              const float add = (image[inputIndex] + image[neighbourIndex]);
              const float add_3 = (image[inputIndex] + 3 * image[neighbourIndex]);
              double current = weights[weightsIndex] * (diff * (gamma * diff_abs + add_3))
                               / ((add + gamma * diff_abs + epsilon) * (add + gamma * diff_abs + epsilon));
              if (do_kappa)
                {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              voxel_gradient += current;
            }
        }
    }
  tmp_grad[inputIndex] = penalisation_factor * voxel_gradient;
}

extern "C" __global__ void
computeCudaRelativeDifferencePriorValueKernel(double* tmp_value,
                                              const float* image,
                                              const float* weights,
                                              const float* kappa,
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

              const float diff = (image[inputIndex] - image[neighbourIndex]);
              const float add = (image[inputIndex] + image[neighbourIndex]);
              double current = (weights[weightsIndex] * 0.5 * diff * diff) / (add + gamma * abs(diff) + epsilon);
              if (do_kappa)
                {
                  current *= kappa[inputIndex] * kappa[neighbourIndex];
                }
              sum += current;
            }
        }
    }
  tmp_value[inputIndex] = penalisation_factor * sum;
}

START_NAMESPACE_STIR

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

  const DiscretisedDensityOnCartesianGrid<3, elemT>& current_image_cast
      = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, elemT>&>(current_image_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaRelativeDifferencePrior: set_up_cuda has not been called\n");
    }

  if (this->weights.get_length() == 0)
    {
      compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
    }
  auto kappa_ptr = this->get_kappa_sptr();
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("RelativeDifferencePrior: kappa image has not the same index range as the reconstructed image\n");

  const int z_dim = this->z_dim;
  const int y_dim = this->y_dim;
  const int x_dim = this->x_dim;

  std::vector<float> image_data(current_image_estimate.size_all());
  std::vector<float> weights_data(this->weights.size_all());
  std::copy(current_image_estimate.begin_all(), current_image_estimate.end_all(), image_data.begin());
  std::copy(this->weights.begin_all(), this->weights.end_all(), weights_data.begin());

  float *d_image_data, *d_weights_data, *d_gradient_data;

  // Allocate memory on the GPU
  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(float));
  cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float));
  cudaMalloc(&d_gradient_data, prior_gradient.size_all() * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_image_data, image_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights_data, weights_data.data(), this->weights.size_all() * sizeof(float), cudaMemcpyHostToDevice);

  if (do_kappa)
    {
      std::vector<float> kappa_data(current_image_estimate.size_all());
      std::copy(kappa_ptr->begin_all(), kappa_ptr->end_all(), kappa_data.begin());
      float* d_kappa;
      cudaMalloc(&d_kappa, current_image_estimate.size_all() * sizeof(float));
      cudaMemcpy(d_kappa, kappa_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
      computeCudaRelativeDifferencePriorGradientKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_gradient_data,
                                                                                          d_image_data,
                                                                                          d_weights_data,
                                                                                          d_kappa,
                                                                                          do_kappa,
                                                                                          this->gamma,
                                                                                          this->epsilon,
                                                                                          this->penalisation_factor,
                                                                                          z_dim,
                                                                                          y_dim,
                                                                                          x_dim);
      cudaFree(d_kappa);
    }
  else
    {
      computeCudaRelativeDifferencePriorGradientKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_gradient_data,
                                                                                          d_image_data,
                                                                                          d_weights_data,
                                                                                          nullptr,
                                                                                          do_kappa,
                                                                                          this->gamma,
                                                                                          this->epsilon,
                                                                                          this->penalisation_factor,
                                                                                          z_dim,
                                                                                          y_dim,
                                                                                          x_dim);
    }

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_weights_data);
      cudaFree(d_gradient_data);
      // error("CUDA error in compute_value kernel execution: " << cudaGetErrorString(cuda_error));
    }

  // Allocate host memory for the result and copy from device to host
  std::vector<float> gradient_data(current_image_estimate.size_all());
  cudaMemcpy(gradient_data.data(), d_gradient_data, current_image_estimate.size_all() * sizeof(float), cudaMemcpyDeviceToHost);

  // Copy the gradient data to the prior_gradient
  std::copy(gradient_data.begin(), gradient_data.end(), prior_gradient.begin_all());

  // Cleanup
  cudaFree(d_image_data);
  cudaFree(d_weights_data);
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

  const DiscretisedDensityOnCartesianGrid<3, elemT>& current_image_cast
      = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, elemT>&>(current_image_estimate);

  if (this->_already_set_up == false)
    {
      error("CudaRelativeDifferencePrior: set_up_cuda has not been called\n");
    }

  if (this->weights.get_length() == 0)
    {
      compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
    }
  // Need to get the kappa image
  auto kappa_ptr = this->get_kappa_sptr();
  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("RelativeDifferencePrior: kappa image has not the same index range as the reconstructed image\n");

  // Assuming z_dim, y_dim, and x_dim are correctly set
  const int z_dim = this->z_dim;
  const int y_dim = this->y_dim;
  const int x_dim = this->x_dim;

  std::vector<float> image_data(current_image_estimate.size_all());
  std::vector<float> weights_data(this->weights.size_all());
  std::copy(current_image_estimate.begin_all(), current_image_estimate.end_all(), image_data.begin());
  std::copy(this->weights.begin_all(), this->weights.end_all(), weights_data.begin());

  // GPU memory pointers
  float *d_image_data, *d_weights_data;
  double* d_tmp_value;

  // Allocate memory on the GPU
  cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(float));
  cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float)); // Assuming weights is also a flat vector
  cudaMalloc(&d_tmp_value, current_image_estimate.size_all() * sizeof(double));

  // Copy data from host to device
  cudaMemcpy(d_image_data, image_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights_data, weights_data.data(), this->weights.size_all() * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  if (do_kappa)
    {
      std::vector<float> kappa_data(current_image_estimate.size_all());
      std::copy(kappa_ptr->begin_all(), kappa_ptr->end_all(), kappa_data.begin());
      float* d_kappa;
      cudaMalloc(&d_kappa, current_image_estimate.size_all() * sizeof(float));
      cudaMemcpy(d_kappa, kappa_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
      computeCudaRelativeDifferencePriorValueKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_tmp_value,
                                                                                       d_image_data,
                                                                                       d_weights_data,
                                                                                       d_kappa,
                                                                                       do_kappa,
                                                                                       this->gamma,
                                                                                       this->epsilon,
                                                                                       this->penalisation_factor,
                                                                                       z_dim,
                                                                                       y_dim,
                                                                                       x_dim);
      cudaFree(d_kappa);
    }
  else
    {
      computeCudaRelativeDifferencePriorValueKernel<<<cuda_grid_dim, cuda_block_dim>>>(d_tmp_value,
                                                                                       d_image_data,
                                                                                       d_weights_data,
                                                                                       nullptr,
                                                                                       do_kappa,
                                                                                       this->gamma,
                                                                                       this->epsilon,
                                                                                       this->penalisation_factor,
                                                                                       z_dim,
                                                                                       y_dim,
                                                                                       x_dim);
    }

  // Check for any errors during kernel execution
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      cudaFree(d_image_data);
      cudaFree(d_weights_data);
      cudaFree(d_tmp_value);
      // error("CUDA error in compute_value kernel execution: " << cudaGetErrorString(cuda_error));
      return 0.0; // Handle error appropriately
    }

  // Allocate host memory for the result and copy from device to host
  std::vector<double> tmp_value(current_image_estimate.size_all());
  cudaMemcpy(tmp_value.data(), d_tmp_value, current_image_estimate.size_all() * sizeof(double), cudaMemcpyDeviceToHost);

  // Compute the total value from tmp_value if necessary
  double totalValue = std::accumulate(tmp_value.begin(), tmp_value.end(), 0.0);

  // Cleanup
  cudaFree(d_image_data);
  cudaFree(d_weights_data);
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
  const DiscretisedDensityOnCartesianGrid<3, float>& target_cast
      = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>&>(*target_sptr);

  this->z_dim = target_cast.get_max_index() - target_cast.get_min_index() + 1;
  this->y_dim = target_cast[0].get_max_index() - target_cast[0].get_min_index() + 1;
  this->x_dim = target_cast[0][0].get_max_index() - target_cast[0][0].get_min_index() + 1;

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
  compute_weights(this->weights, target_cast.get_grid_spacing(), this->only_2D);
  this->_already_set_up = true;
  return Succeeded::yes;
}

template class CudaRelativeDifferencePrior<float>;

END_NAMESPACE_STIR
