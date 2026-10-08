#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUBackwardProjectorCUDA.h"

#include <cuda_runtime.h>

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPURotateAndGaussianInterpolate.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUPSFGaussian.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"
#include <cmath>

START_NAMESPACE_STIR

__global__ void
add_arrays(float* dst, const float* src, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    dst[i] += src[i];
}

void
accumulate_image_omp_contrib(float* dev_image,
                             const float* thread_image,
                             unsigned int block_x,
                             unsigned int block_y,
                             unsigned int block_z,
                             unsigned int grid_x,
                             unsigned int grid_y,
                             unsigned int grid_z,
                             unsigned int image_size)
{

  const int threads = block_x * block_y * block_z;
  const int blocks = (image_size + threads - 1) / threads;

  add_arrays<<<blocks, threads>>>(dev_image, thread_image, image_size);

  auto err0 = cudaGetLastError();
  if (err0 != cudaSuccess)
    error(cudaGetErrorString(err0));
}

void
initialise_im_buffers(AllocatedStack& stack, bool do_atten, bool do_density)
{
  if (do_density)
    {
      cudaMemset(stack.dev_image.data(), 0, stack.image_size * sizeof(float));
    }

  if (do_atten)
    {
      cudaMemset(stack.dev_umap.data(), 0, stack.image_size * sizeof(float));
    }
}
void
run_backward_projection_cuda(AllocatedStack& stack,
                             const RelatedViewgrams<float>& stir_sino,
                             bool do_atten,
                             float coll_sigma0_cm,
                             float num_sigmas,
                             float coll_slope,
                             int num_views,
                             unsigned int block_x,
                             unsigned int block_y,
                             unsigned int block_z,
                             unsigned int grid_x,
                             unsigned int grid_y,
                             unsigned int grid_z,
                             float spacing_x,
                             float spacing_y,
                             float spacing_z,
                             float origin_x,
                             float origin_y,
                             float origin_z,
                             int dim_x,
                             int dim_y,
                             int dim_z,
                             int min_z,
                             int min_y,
                             int min_x)
{
  //    cudaDeviceSynchronize();
  //    std::cout << "ENTER BP CUDA" << std::endl;
  dim3 cuda_block_dim(block_x, block_y, block_z);

  dim3 cuda_grid_dim(grid_x, grid_y, grid_z);

  float3 spacing = make_float3(spacing_x, spacing_y, spacing_z);

  float3 origin = make_float3(origin_x, origin_y, origin_z);

  int3 image_dim = make_int3(dim_x, dim_y, dim_z);
  int3 min_indices = make_int3(min_x, min_y, min_z);

  auto vg_iter = stir_sino.begin();
  const Viewgram<float>& vg = *vg_iter;

  if (!stack.is_allocated())
    error("SPECTGPUBP: Something is wrong the CuVecs are not initialised");

  Bin bin(0, vg.get_view_num(), 0, 0, 0);
  // the following sign is introduced to match SPECTUB
  float angle_rad = vg.get_proj_data_info().get_phi(
      bin); //-vg.get_view_num() * 2.f * M_PI / num_views;//vg.get_proj_data_info().get_phi(); //
  angle_rad = -std::fmod(angle_rad, 2.f * M_PI);
  // Note that umap is rotated before because the BP needs to apply attenuation factors corresponding to the same rotation as the
  // image

  //  reinitialise evrything
  cudaMemset(stack.rotated_umap.data(), 0, dim_x * dim_y * dim_z * sizeof(float));
  cudaMemset(stack.rotated_im.data(), 0, dim_x * dim_y * dim_z * sizeof(float));
  array_to_device(stack.dev_sino, vg);
  //  cudaMemset(stack.rotated_im.data(), 0, dim_x * dim_y * dim_z * sizeof(float));
  cudaMemset(stack.blurred_im.data(), 0, dim_x * dim_y * dim_z * sizeof(float));

  if (do_atten)
    {
      rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
          stack.rotated_umap.data(), stack.dev_umap.data(), image_dim, spacing, origin, min_indices, angle_rad);

      auto err0 = cudaGetLastError();
      if (err0 != cudaSuccess)
        error(cudaGetErrorString(err0));
    }

  // Actual BP
  backwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(
      stack.rotated_im.data(), stack.dev_sino.data(), stack.rotated_umap.data(), image_dim, spacing, do_atten);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    error(cudaGetErrorString(err));

  //        PSF
  if (coll_sigma0_cm >= 0 && coll_slope >= 0)
    {
      GaussianConvolutionKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          stack.blurred_im.data(), stack.rotated_im.data(), image_dim, spacing, coll_sigma0_cm, num_sigmas, coll_slope);

      auto errpsf_f0 = cudaGetLastError();
      if (errpsf_f0 != cudaSuccess)
        error(cudaGetErrorString(errpsf_f0));

      // Rotation+Interpolation
      rotateKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          stack.dev_image.data(), stack.blurred_im.data(), image_dim, spacing, origin, min_indices, angle_rad);

      auto errpsf_r = cudaGetLastError();
      if (errpsf_r != cudaSuccess)
        error(cudaGetErrorString(errpsf_r));
    }
  else
    {
      // Rotation+Interpolation
      rotateKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          stack.dev_image.data(), stack.rotated_im.data(), image_dim, spacing, origin, min_indices, angle_rad);

      auto err1 = cudaGetLastError();
      if (err1 != cudaSuccess)
        error(cudaGetErrorString(err1));
    }
}

END_NAMESPACE_STIR
