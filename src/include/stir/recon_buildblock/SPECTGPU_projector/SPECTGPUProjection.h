#pragma once

#include <cuda_runtime.h>

__global__
void forwardKernel(
    const float* image,
    float* sino,
    int3 image_dim);

__global__
void backwardKernel(
        const float* sino,
        float* image,
        int3 image_di,
        float3 spacing);
