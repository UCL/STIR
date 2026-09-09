#pragma once

#include <cuda_runtime.h>

__global__
void rotateKernel_pull(
        const float* in_im,
        float* out_im,
        int3 image_dim,
        float3 spacing,
        float3 origin,
        int3 min_indeces,
        float angle_rad);

__global__
void rotateKernel_push(
        const float* in_im,
        float* out_im,
        int3 image_dim,
        float3 spacing,
        float3 origin,
        int3 min_indeces,
        float angle_rad);
