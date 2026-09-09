#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUForwardProjectorCUDA.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <cuda_runtime.h>

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPURotateAndGaussianInterpolate.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"

START_NAMESPACE_STIR

void run_forward_projection_cuda(
        RelatedViewgrams<float>& stir_sino,
        const DiscretisedDensity<3,float>& stir_image,
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
    //for all views in relateViewgram call the kernels
    dim3 cuda_block_dim(block_x, block_y, block_z);
    dim3 cuda_grid_dim(grid_x, grid_y, grid_z);
//    viewgrams = _projected_data_sptr->get_related_viewgrams(stir_sino.get_basic_view_segment_num(), _symmetries_sptr);

    float* dev_image;
    cudaMalloc(&dev_image, stir_image.size_all() * sizeof(float));
    float* out_im;
    cudaMalloc(&out_im, stir_image.size_all() * sizeof(float));
     auto& vox =
        dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir_image);

    array_to_device(dev_image, vox);

//    array_to_device(dev_image, stir_image);


    float3 spacing = make_float3(spacing_x,
                                 spacing_y,
                                 spacing_z);

    float3 origin = make_float3(origin_x,
                                origin_y,
                                origin_z
                                );

    int3 min_indeces = make_int3(min_x, min_y, min_z);

    int3 image_dim = make_int3(dim_x, dim_y, dim_z);

//    const int num_views = stir_sino.get_num_viewgrams();

    float* dev_sino = nullptr;
//    Viewgram<float>& vg0 = stir_sino.[0];
    auto vg_iter = stir_sino.begin();

    Viewgram<float>& vg0 = *vg_iter;
    const auto sino_size = vg0.size_all();
    int dim_ax = vg0.get_num_axial_poss();
    int dim_tg = vg0.get_num_tangential_poss();
    cudaMalloc(&dev_sino,
               sino_size*sizeof(float));


    if (vg0.size_all() != dim_ax * dim_tg)
      error("SPECTGPU: Viewgram size does not match kernel output size.");

    if (vg0.get_num_axial_poss() != dim_ax)
      error("SPECTGPU: Viewgram axial dimension does not match image z dimension.");

    if (vg0.get_num_tangential_poss() != dim_tg)
      error("SPECTGPU: Viewgram tangential dimension does not match image x dimension.");


    for (auto vg_iter = stir_sino.begin();
         vg_iter != stir_sino.end();
         ++vg_iter)
    {
        Viewgram<float>& vg = *vg_iter;
//        std::cout
//            << "view counter = " << view
//            << "  stir view = " << vg.get_view_num()
//            << std::endl;
        //the following sign is introduced to match SPECTUB
        float angle_rad = -vg.get_view_num() * 2.f * M_PI / num_views;

//        std::cout<<"view and angle = "<<vg.get_view_num()<<" "<<angle_rad<<std::endl;

        rotateKernel_pull<<<cuda_grid_dim, cuda_block_dim>>>(
                                                               dev_image,
                                                               out_im,
                                                               image_dim,
                                                               spacing,
                                                               origin,
                                                               min_indeces,
                                                               angle_rad);

        cudaDeviceSynchronize();

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            error(cudaGetErrorString(err));

        //array_to_device(dev_sino, vg); don't need this asthe viewgrams need to be filled by the kernel
        //so need to set everything to zero

        cudaMemset(dev_sino,
                   0,
                   sino_size*sizeof(float));

        forwardKernel<<<cuda_grid_dim, cuda_block_dim>>>(
                                                           out_im,
                                                           dev_sino,
                                                           image_dim);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
            error(cudaGetErrorString(err));

        array_to_host(vg, dev_sino);

      }
    cudaFree(dev_image);
    cudaFree(out_im);
    cudaFree(dev_sino);
      //  cudaMalloc(&cuda_image, stir_image_sptr->size_all() * sizeof(elemT));
    //  array_to_device(cuda_image, *stir_image_sptr);
    }


END_NAMESPACE_STIR
