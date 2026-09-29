// CUDA kernels for element-wise operations on ProjData

#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInMemoryCUDA.h"
#include "cuvec.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

// Not the cleanest way, useful for testing
#ifndef NUMCU_THREADS
#define NUMCU_THREADS 512
#endif
//

START_NAMESPACE_STIR

// Functors of the operators
struct AddAssignOp {
    __device__ float operator()(float lhs, float rhs) const {
        return lhs + rhs;
    }
};

struct SubAssignOp {
    __device__ float operator()(float lhs, float rhs) const {
        return lhs - rhs;
    }
};

struct MultAssignOp {
    __device__ float operator()(float lhs, float rhs) const {
        return lhs * rhs;
    }
};

struct DivAssignOp {
    __device__ float operator()(float lhs, float rhs) const {
        return lhs / rhs;
    }
};

// Kernels template
template <class Op>
__global__
void knlBinaryAssign(float* dst,
                     const float* src,
                     size_t N)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        Op op;
        dst[i] = op(dst[i], src[i]);
    }
}

// Generalized Wrapper
template <class Op>
void BinaryAssign(float* dst,
                  const float* src,
                  size_t N)
{
    dim3 threads(NUMCU_THREADS, 1, 1);
    dim3 blocks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);

    knlBinaryAssign<Op><<<blocks, threads>>>(dst, src, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    cudaDeviceSynchronize();
}

// Public interface
void AddAssign(float* dst,
               const float* src,
               size_t N)
{
    BinaryAssign<AddAssignOp>(dst, src, N);
}

void SubAssign(float* dst,
               const float* src,
               size_t N)
{
    BinaryAssign<SubAssignOp>(dst, src, N);
}

void MultAssign(float* dst,
                const float* src,
                size_t N)
{
    BinaryAssign<MultAssignOp>(dst, src, N);
}

void DivAssign(float* dst,
               const float* src,
               size_t N)
{
    BinaryAssign<DivAssignOp>(dst, src, N);
}

// XAPYB operator
// Kernel version for xapyb
__global__ void knlxapyb(float *dst,
                       const float *x,
                       const float *y,
                       float a,
                       float b,
                       const size_t N)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    
    dst[i] = (x[i]) * a + (y[i]) * b;

}

// Wrapper xapyb
void CUDAxapyb(float *dst, 
               const float *x,
               const float *y,
               float a,
               float b, 
               const size_t N) 

{

    dim3 threads(NUMCU_THREADS, 1, 1);
    dim3 blocks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);

    knlxapyb<<<blocks, threads>>>(dst, x, y, a, b, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));

    cudaDeviceSynchronize();

}

END_NAMESPACE_STIR