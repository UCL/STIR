/*
    Copyright (C) 2024, University College London
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
*/
#include "stir/Array.h"
#include "stir/info.h"
#include "stir/error.h"
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

#ifdef __CUDACC__
template <int num_dimensions, typename elemT>
inline void
array_to_device(elemT* dev_data, const Array<num_dimensions, elemT>& stir_array)
{
  if (stir_array.is_contiguous())
    {
      info("array_to_device contiguous", 100);
      cudaMemcpy(dev_data, stir_array.get_const_full_data_ptr(), stir_array.size_all() * sizeof(elemT), cudaMemcpyHostToDevice);
      stir_array.release_const_full_data_ptr();
    }
  else
    {
      info("array_to_device non-contiguous", 100);
      // Allocate host memory to get contiguous vector, copy array to it and copy from device to host
      std::vector<elemT> tmp_data(stir_array.size_all());
      std::copy(stir_array.begin_all(), stir_array.end_all(), tmp_data.begin());
      cudaMemcpy(dev_data, tmp_data.data(), stir_array.size_all() * sizeof(elemT), cudaMemcpyHostToDevice);
    }
}

template <int num_dimensions, typename elemT>
inline void
array_to_host(Array<num_dimensions, elemT>& stir_array, const elemT* dev_data)
{
  if (stir_array.is_contiguous())
    {
      info("array_to_host contiguous", 100);
      cudaMemcpy(stir_array.get_full_data_ptr(), dev_data, stir_array.size_all() * sizeof(elemT), cudaMemcpyDeviceToHost);
      stir_array.release_full_data_ptr();
    }
  else
    {
      info("array_to_host non-contiguous", 100);
      // Allocate host memory for the result and copy from device to host
      std::vector<elemT> tmp_data(stir_array.size_all());
      cudaMemcpy(tmp_data.data(), dev_data, stir_array.size_all() * sizeof(elemT), cudaMemcpyDeviceToHost);
      // Copy the data to the stir_array
      std::copy(tmp_data.begin(), tmp_data.end(), stir_array.begin_all());
    }
}

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
  double dval = static_cast<double>(val);
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
