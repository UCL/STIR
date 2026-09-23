/*
    Copyright (C) 2024, 2026, University College London
    Copyright (C) 2025, University of Milano-Bicocca
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_cuda_utilities_H__
#define __stir_cuda_utilities_H__

/*!
  \file
  \ingroup CUDA
  \brief some utilities for STIR and CUDA

  \author Kris Thielemans
  \author Matteo Neel Colombo
*/
#include "stir/Array.h"
#include "stir/info.h"
#include "stir/error.h"
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#  include "cuvec.cuh"
#endif
#include <vector>

START_NAMESPACE_STIR

#ifndef __CUDACC__
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __device__
#    define __device__
#  endif
#endif

#ifndef __CUDACC__
struct cuda_dim3
{
  unsigned int x = 1, y = 1, z = 1;
};
struct cuda_int3
{
  int x = 0, y = 0, z = 0;
};
#else
typedef dim3 cuda_dim3;
typedef int3 cuda_int3;
#endif

#ifdef __CUDACC__

//! copy an `Array` to pre-allocated device memory
/*!
  \ingroup CUDA
*/
template <int num_dimensions, typename elemT>
void array_to_device(elemT* dev_data, const Array<num_dimensions, elemT>& stir_array);

//! copy an `Array` to pre-allocated CuVec
/*!
  \ingroup CUDA
*/
template <int num_dimensions, typename elemT>
void array_to_device(CuVec<elemT>& dev_data, const Array<num_dimensions, elemT>& stir_array);

//! copy CUDA pointer to `Array`
/*!
  \ingroup CUDA
  The third argument is ignored, as `cudaMemcpy` always syncs device and host.
*/
template <int num_dimensions, typename elemT>
void array_to_host(Array<num_dimensions, elemT>& stir_array, const elemT* dev_data, bool /* sync */ = true);
//! copy CuVec to `Array`
/*!
  \ingroup CUDA
  If \a sync = \c true, the function will call `cudaDeviceSynchronize()` before copying.
*/
template <int num_dimensions, typename elemT>
void array_to_host(Array<num_dimensions, elemT>& stir_array, const CuVec<elemT>& dev_data, bool sync = true);

//! \brief Performs a parallel reduction sum on shared memory within a CUDA thread block, final value stored in shared_mem[0].
template <typename elemT>
__device__ inline void
blockReduction(elemT* shared_mem, int thread_in_block, int block_threads)
{
  for (int stride = block_threads / 2; stride > 0; stride /= 2)
    {
      if (thread_in_block < stride)
        shared_mem[thread_in_block] += shared_mem[thread_in_block + stride];
      __syncthreads();
    }
}

//! \brief Provides atomic addition for double values with fallback for pre-Pascal GPU architectures.
template <typename elemT>
__device__ inline double
atomicAddGeneric(double* address, elemT val)
{
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(address, static_cast<double>(val));
#  else
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    {
      printf("CudaGibbsPenalty: atomicAdd(double) unsupported on this GPU. "
             "Upgrade to compute capability >= 6.0 or check code at "
             "sources/STIR/src/include/stir/cuda_utilities.h:108.\n");
      asm volatile("trap;");
    }
  return 0.0; // never reached
              // Emulate atomicAdd for double precision on pre-Pascal architectures
              // unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
              // unsigned long long int old = *address_as_ull, assumed;

  // do
  //   {
  //     assumed = old;
  //     double updated = __longlong_as_double(assumed) + dval;
  //     old = atomicCAS(address_as_ull, assumed, __double_as_longlong(updated));
  // } while (assumed != old);

  // return __longlong_as_double(old);
#  endif
}

//! \brief Utility function to check for CUDA errors and report them with context information.
inline void
checkCudaError(const std::string& operation)
{
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
    {
      const char* err = cudaGetErrorString(cuda_error);
      error(std::string("CudaGibbsPrior: CUDA error in ") + operation + ": " + err);
    }
}
#endif

END_NAMESPACE_STIR
#endif
