#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUBackwardProjectorCUDA.h"

#include <cuda_runtime.h>

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPURotateAndGaussianInterpolate.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUPSFGaussian.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"
#include <cmath>

START_NAMESPACE_STIR

void
allocate_im_buffers(float*& dev_image,
                    float*& dev_umap,
                    const DiscretisedDensity<3, float>& image,
                    const DiscretisedDensity<3, float>& umap,
                    bool do_atten)
{
  cudaMalloc((void**)&dev_image, image.size_all() * sizeof(float));

  cudaMemset(dev_image, 0, image.size_all() * sizeof(float));
  if (do_atten)
    {
      cudaMalloc((void**)&dev_umap, umap.size_all() * sizeof(float));

      cudaMemset(dev_umap, 0, umap.size_all() * sizeof(float));
    }
}

void
free_im_buffers(float* dev_image, float* dev_umap, bool do_atten)
{
  cudaFree(dev_image);
  if (do_atten)
    cudaFree(dev_umap);
}

void
copy_im_to_stir(DiscretisedDensity<3, float>& image, const float* dev_image)
{
  array_to_host(image, dev_image);
}

void
copy_stir_im_to_dev(float* dev_image, const DiscretisedDensity<3, float>& image)
{
  array_to_device(dev_image, image);
}

void
run_backward_projection_cuda(float* dev_image,
                             const RelatedViewgrams<float>& stir_sino,
                             const float* dev_umap,
                             bool do_atten,
                             float coll_sigma0_cm,
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
  //    std::cout << "ENTER BP CUDA" << std::endl;
  dim3 cuda_block_dim(block_x, block_y, block_z);

  dim3 cuda_grid_dim(grid_x, grid_y, grid_z);

  CuVec<float> rotated_im(dim_x * dim_y * dim_z);

  CuVec<float> rotated_umap(dim_x * dim_y * dim_z);
  //    float* dev_umap;

  float3 spacing = make_float3(spacing_x, spacing_y, spacing_z);

  float3 origin = make_float3(origin_x, origin_y, origin_z);

  int3 image_dim = make_int3(dim_x, dim_y, dim_z);
  int3 min_indices = make_int3(min_x, min_y, min_z);

  CuVec<float> blurred_im(dim_x * dim_y * dim_z);

  auto vg_iter = stir_sino.begin();
  const Viewgram<float>& vg = *vg_iter;
  const auto sino_size = vg.size_all();

  Bin bin(0, vg.get_view_num(), 0, 0, 0);
  // the following sign is introduced to match SPECTUB
  float angle_rad = vg.get_proj_data_info().get_phi(
      bin); //-vg.get_view_num() * 2.f * M_PI / num_views;//vg.get_proj_data_info().get_phi(); //
  angle_rad = -std::fmod(angle_rad, 2.f * M_PI);

  if (do_atten)
    {
      rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
          rotated_umap.data(), dev_umap, image_dim, spacing, origin, min_indices, angle_rad);

      cudaDeviceSynchronize();

      auto err0 = cudaGetLastError();
      if (err0 != cudaSuccess)
        error(cudaGetErrorString(err0));
    }

  CuVec<float> dev_sino(sino_size);

  array_to_device(dev_sino, vg);

  cudaMemset(rotated_im.data(), 0, dim_x * dim_y * dim_z * sizeof(float));

  // Actual BP
  backwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(
      rotated_im.data(), dev_sino.data(), rotated_umap.data(), image_dim, spacing, do_atten);

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
    error(cudaGetErrorString(err));

  //        PSF
  if (coll_sigma0_cm >= 0 && coll_slope >= 0)
    {

      cudaMemset(blurred_im.data(), 0, dim_x * dim_y * dim_z * sizeof(float));

      GaussianConvolutionKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          blurred_im.data(), rotated_im.data(), image_dim, spacing, coll_sigma0_cm, coll_slope);

      auto errpsf_f0 = cudaGetLastError();
      if (errpsf_f0 != cudaSuccess)
        error(cudaGetErrorString(errpsf_f0));

      // Rotation+Interpolation
      rotateKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          dev_image, blurred_im.data(), image_dim, spacing, origin, min_indices, angle_rad);

      auto errpsf_r = cudaGetLastError();
      if (errpsf_r != cudaSuccess)
        error(cudaGetErrorString(errpsf_r));
    }
  else
    {
      // Rotation+Interpolation
      rotateKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
          dev_image, rotated_im.data(), image_dim, spacing, origin, min_indices, angle_rad);

      auto err1 = cudaGetLastError();
      if (err1 != cudaSuccess)
        error(cudaGetErrorString(err1));
    }
}

END_NAMESPACE_STIR
