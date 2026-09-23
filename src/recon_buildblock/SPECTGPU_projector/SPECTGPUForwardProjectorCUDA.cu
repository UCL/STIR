#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUForwardProjectorCUDA.h"

#include "stir/VoxelsOnCartesianGrid.h"
//#include <cuda_runtime.h>

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPURotateAndGaussianInterpolate.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUPSFGaussian.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"
#include "stir/Bin.h"
#include <cmath>

START_NAMESPACE_STIR
#ifdef __CUDACC__
void
run_forward_projection_cuda(RelatedViewgrams<float>& stir_sino,
                            float* dev_image,
                            const float* dev_umap,
                            bool do_atten,
                            float coll_sigma0_cm,
                            float coll_slope,
                            int num_views,
                            int min_ax,
                            int max_ax,
                            int min_tg,
                            int max_tg,
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
  // for all views in relateViewgram call the kernels
  dim3 cuda_block_dim(block_x, block_y, block_z);
  dim3 cuda_grid_dim(grid_x, grid_y, grid_z);

  CuVec<float> out_im(dim_x * dim_y * dim_z);
  CuVec<float> out_umap(dim_x * dim_y * dim_z);

  float3 spacing = make_float3(spacing_x, spacing_y, spacing_z);

  float3 origin = make_float3(origin_x, origin_y, origin_z);

  int3 min_indices = make_int3(min_x, min_y, min_z);

  int3 image_dim = make_int3(dim_x, dim_y, dim_z);

  auto vg_iter = stir_sino.begin();

  Viewgram<float>& vg0 = *vg_iter;
  const auto sino_size = vg0.size_all();
  CuVec<float> dev_sino(sino_size);
  int dim_ax = vg0.get_num_axial_poss();
  int dim_tg = vg0.get_num_tangential_poss();

  //    float* blurred_im;
  CuVec<float> blurred_im(dim_x * dim_y * dim_z);
  //    if(coll_sigma0_cm>=0 && coll_slope>=0)
  //        cudaMalloc(&blurred_im, dim_x*dim_y*dim_z * sizeof(float));
  if (vg0.size_all() != dim_ax * dim_tg)
    error("SPECTGPU: Viewgram size does not match kernel output size.");

  if (vg0.get_num_axial_poss() != dim_ax)
    error("SPECTGPU: Viewgram axial dimension does not match image z dimension.");

  if (vg0.get_num_tangential_poss() != dim_tg)
    error("SPECTGPU: Viewgram tangential dimension does not match image x dimension.");

  for (auto vg_iter = stir_sino.begin(); vg_iter != stir_sino.end(); ++vg_iter)
    {
      Viewgram<float>& vg = *vg_iter;
      Bin bin(0, vg.get_view_num(), 0, 0, 0);
      // the following sign is introduced to match SPECTUB
      float angle_rad = vg.get_proj_data_info().get_phi(
          bin); //-vg.get_view_num() * 2.f * M_PI / num_views;//vg.get_proj_data_info().get_phi(); //
      angle_rad = -std::fmod(angle_rad, 2.f * M_PI);
      //        if (angle_rad >2.f * M_PI)
      //            angle_rad -=2.f * M_PI;

      //        std::cout<<"view and angle = "<<-vg.get_view_num()* 2.f * M_PI / num_views<<" "<<angle_rad<<"
      //        "<<vg.get_view_num()<<std::endl;

      if (do_atten)
        {
          rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
              out_umap.data(), dev_umap, image_dim, spacing, origin, min_indices, angle_rad);

          auto err0 = cudaGetLastError();
          if (err0 != cudaSuccess)
            error(cudaGetErrorString(err0));
        }

      rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
          out_im.data(), dev_image, image_dim, spacing, origin, min_indices, angle_rad);

      auto err = cudaGetLastError();
      if (err != cudaSuccess)
        error(cudaGetErrorString(err));

      if (coll_sigma0_cm >= 0 && coll_slope >= 0)
        {

          //            cudaMalloc(&dev_umap, stir_image.size_all() * sizeof(float));
          //            array_to_device(blurr_im, stir_umap);
          GaussianConvolutionKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
              blurred_im.data(), out_im.data(), image_dim, spacing, coll_sigma0_cm, coll_slope);

          auto errpsf_f0 = cudaGetLastError();
          if (errpsf_f0 != cudaSuccess)
            error(cudaGetErrorString(errpsf_f0));

          cudaMemset(dev_sino.data(), 0, sino_size * sizeof(float));

          forwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(
              dev_sino.data(), blurred_im.data(), out_umap.data(), image_dim, spacing, do_atten);

          auto errpsf_f = cudaGetLastError();
          if (errpsf_f != cudaSuccess)
            error(cudaGetErrorString(errpsf_f));
        }
      else
        {

          // array_to_device(dev_sino, vg); don't need this asthe viewgrams need to be filled by the kernel
          // so need to set everything to zero

          cudaMemset(dev_sino.data(), 0, sino_size * sizeof(float));

          forwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(
              dev_sino.data(), blurred_im.data(), out_umap.data(), image_dim, spacing, do_atten);

          err = cudaGetLastError();
          if (err != cudaSuccess)
            error(cudaGetErrorString(err));
        }

      array_to_host(vg, dev_sino, true);
    }
}
#endif

END_NAMESPACE_STIR
