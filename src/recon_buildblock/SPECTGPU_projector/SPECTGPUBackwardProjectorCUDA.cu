#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUBackwardProjectorCUDA.h"

#include <cuda_runtime.h>

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPURotateAndGaussianInterpolate.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUProjection.h"

START_NAMESPACE_STIR

void run_backward_projection_cuda(
        const RelatedViewgrams<float>& stir_sino,
        DiscretisedDensity<3,float>& stir_image,
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
    dim3 cuda_block_dim(
        block_x,
        block_y,
        block_z);

    dim3 cuda_grid_dim(
        grid_x,
        grid_y,
        grid_z);

    float* dev_image;
    cudaMalloc(
        &dev_image,
        stir_image.size_all() * sizeof(float));

//    cudaMemset(
//        dev_image,
//        0,
//        stir_image.size_all() * sizeof(float));
//    this is different than Forward as STIR calls actual_backproject() for every view
    array_to_device(dev_image, stir_image);


    float* rotated_im;
    cudaMalloc(
        &rotated_im,
        stir_image.size_all() * sizeof(float));

    float3 spacing = make_float3(spacing_x,
                                 spacing_y,
                                 spacing_z);

    float3 origin = make_float3(origin_x,
                                origin_y,
                                origin_z
                                );

    int3 image_dim = make_int3(dim_x, dim_y, dim_z);
    int3 min_indeces = make_int3(min_x, min_y, min_z);


    auto vg_iter = stir_sino.begin();
    //    const Viewgram<float>& vg0 = *vg_iter;
    //    const auto sino_size = vg0.size_all();

//    for (auto vg_iter = stir_sino.begin();
//         vg_iter != stir_sino.end();
//         ++vg_iter)
//    {
        const Viewgram<float>& vg = *vg_iter;
        const auto sino_size = vg.size_all();


        float angle_rad = -vg.get_view_num() * 2.f * M_PI / num_views;

//        if (vg.size_all() != image_dim.x * image_dim.z)
//          error("SPECTGPU: Viewgram size does not match kernel output size.");

//        if (vg.get_num_axial_poss() != image_dim.z)
//          error("SPECTGPU: Viewgram axial dimension does not match image z dimension.");

//        if (vg.get_num_tangential_poss() != image_dim.x)
//          error("SPECTGPU: Viewgram tangential dimension does not match image x dimension.");

        float* dev_sino;

        cudaMalloc(
            &dev_sino,
            sino_size * sizeof(float));

        array_to_device(dev_sino, vg);

        cudaMemset(rotated_im,
                   0,
                   stir_image.size_all() * sizeof(float));

        backwardKernel<<<cuda_grid_dim,cuda_block_dim>>>(
                                                           dev_sino,
                                                           rotated_im,
                                                           image_dim,
                                                           spacing);

        cudaDeviceSynchronize();

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
            error(cudaGetErrorString(err));

//        array_to_host(stir_image, rotated_im);
//        return;

//        std::vector<float> host_im(stir_image.size_all());
//        cudaMemcpy(host_im.data(),
//                   dev_image,
//                   stir_image.size_all() * sizeof(float),
//                   cudaMemcpyDeviceToHost);
//        float sum_before =
//            std::accumulate(host_im.begin(),
//                            host_im.end(),
//                            0.0f);

        rotateKernel_push<<<cuda_grid_dim, cuda_block_dim>>>(
                                                               rotated_im,
                                                               dev_image,
                                                               image_dim,
                                                               spacing,
                                                               origin,
                                                               min_indeces,
                                                               angle_rad);

        cudaDeviceSynchronize();

        auto err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
            error(cudaGetErrorString(err1));

//        cudaMemcpy(host_im.data(),
//                   dev_image,
//                   stir_image.size_all() * sizeof(float),
//                   cudaMemcpyDeviceToHost);

//        cudaMemcpy(host_im.data(),
//        dev_image,
//        stir_image.size_all()*sizeof(float),
//        cudaMemcpyDeviceToHost);

//        float sum =
//            std::accumulate(host_im.begin(),
//                            host_im.end(),
//                            0.0f);

//        float maxv =
//            *std::max_element(host_im.begin(),
//                              host_im.end());

//        std::cout
//            << "view = " << vg.get_view_num()
//            << "  sum after = " << sum
//            << "  sum before = " << sum_before
//            << "  max = " << maxv
//            << std::endl;
//        cudaFree(dev_sino);
////    }

    array_to_host(stir_image, dev_image);

    cudaFree(rotated_im);
    cudaFree(dev_image);
}

END_NAMESPACE_STIR
