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
#include <stdexcept>
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#else
#  ifndef CUVEC_DISABLE_CUDA
#    define CUVEC_DISABLE_CUDA
#  endif
#endif
#include "cuvec.cuh"
#include <vector>
#include <memory>
#ifdef STIR_WITH_CUDA
#  include <cuda_runtime.h>
#  include "stir/algebraic_kernels.h"
#endif

START_NAMESPACE_STIR

#ifdef STIR_WITH_CUDA

template <typename T>
inline bool
onGPU(const T* data)
{
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, data);

  if (err != cudaSuccess)
    {
      cudaGetLastError();
      return false;
    }

  switch (attr.type)
    {
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged:
      return true;

    case cudaMemoryTypeHost:
    case cudaMemoryTypeUnregistered:
      return false;

    default:
      throw std::invalid_argument("Unknown CUDA memory type");
    }
}

template <int num_dimensions, typename elemT>
inline bool
onGPU(const Array<num_dimensions, elemT>& arr)
{
  const elemT* ptr = arr.get_const_full_data_ptr();
  bool result = onGPU(ptr);
  arr.release_const_full_data_ptr();
  return result;
}

template <typename T>
inline bool
onGPU(const CuVec<T>& vec)
{
  return onGPU(vec.data());
}
#endif

template <int num_dimensions, typename elemT>
void
copy(Array<num_dimensions, elemT>& out, const Array<num_dimensions, elemT>& in, bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  const bool out_on_gpu = onGPU(out.get_const_data_ptr());
  const bool in_on_gpu = onGPU(in.get_const_data_ptr());

  if ((out_on_gpu || in_on_gpu) && out.is_contiguous() && in.is_contiguous())
    {
      cudaMemcpyKind kind;
      if (out_on_gpu && in_on_gpu)
        kind = cudaMemcpyDeviceToDevice;
      else if (out_on_gpu && !in_on_gpu)
        kind = cudaMemcpyHostToDevice;
      else
        kind = cudaMemcpyDeviceToHost;

      cudaError_t err = cudaMemcpy(out.get_data_ptr(), in.get_const_data_ptr(), out.size_all() * sizeof(elemT), kind);
      if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

      if (cuda_sync)
        cudaDeviceSynchronize();

      return;
    }
#endif

  // Fallback to CPU copy
  std::copy(in.begin_all(), in.end_all(), out.begin_all());
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>&
add_assign(Array<num_dimensions, elemT>& inout, const Array<num_dimensions, elemT>& arg, bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  if (onGPU(inout.get_const_data_ptr()) && inout.is_contiguous() && arg.is_contiguous())
    {
      add_assign(inout.get_data_ptr(), arg.get_const_data_ptr(), inout.size_all());

      if (cuda_sync)
        cudaDeviceSynchronize();

      return inout;
    }
#endif

  // Fallback for CPU memory or non-contiguous data
  inout += arg;
  return inout;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>&
sub_assign(Array<num_dimensions, elemT>& inout, const Array<num_dimensions, elemT>& arg, bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  if (onGPU(inout.get_const_data_ptr()) && inout.is_contiguous() && arg.is_contiguous())
    {
      sub_assign(inout.get_data_ptr(), arg.get_const_data_ptr(), inout.size_all());
      if (cuda_sync)
        cudaDeviceSynchronize();
      return inout;
    }
#endif

  inout -= arg;
  return inout;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>&
mult_assign(Array<num_dimensions, elemT>& inout, const Array<num_dimensions, elemT>& arg, bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  if (onGPU(inout.get_const_data_ptr()) && inout.is_contiguous() && arg.is_contiguous())
    {
      mult_assign(inout.get_data_ptr(), arg.get_const_data_ptr(), inout.size_all());

      if (cuda_sync)
        cudaDeviceSynchronize();

      return inout;
    }
#endif

  // CPU fallback for standard C++ arrays or non-contiguous data
  inout *= arg;
  return inout;
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>&
div_assign(Array<num_dimensions, elemT>& inout, const Array<num_dimensions, elemT>& arg, bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  if (onGPU(inout.get_const_data_ptr()) && inout.is_contiguous() && arg.is_contiguous())
    {
      div_assign(inout.get_data_ptr(), arg.get_const_data_ptr(), inout.size_all());

      if (cuda_sync)
        cudaDeviceSynchronize();

      return inout;
    }
#endif

  // CPU fallback for standard C++ arrays or non-contiguous data
  inout /= arg;
  return inout;
}

template <int num_dimensions, typename elemT>
void
xapyb(Array<num_dimensions, elemT>& dst,
      const Array<num_dimensions, elemT>& x,
      const Array<num_dimensions, elemT>& y,
      const elemT a,
      const elemT b,
      bool cuda_sync = true)
{

#ifdef STIR_WITH_CUDA
  if (onGPU(dst.get_const_data_ptr()) && dst.is_contiguous() && x.is_contiguous() && y.is_contiguous())
    {
      CUDAxapyb(dst.get_data_ptr(), x.get_const_data_ptr(), y.get_const_data_ptr(), a, b, dst.size_all());
      if (cuda_sync)
        cudaDeviceSynchronize();
      return;
    }
#endif

  // CPU fallback
  typename Array<num_dimensions, elemT>::full_iterator dst_it = dst.begin_all();
  typename Array<num_dimensions, elemT>::const_full_iterator x_it = x.begin_all();
  typename Array<num_dimensions, elemT>::const_full_iterator y_it = y.begin_all();

  while (dst_it != dst.end_all())
    {
      *dst_it = (*x_it) * a + (*y_it) * b;
      ++dst_it;
      ++x_it;
      ++y_it;
    }
}

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

#ifdef STIR_WITH_CUDA
namespace detail
{
//! Pointer a kernel can read: the array's own pointer if on the GPU, otherwise a device copy in \a tmp.
template <int num_dimensions, typename elemT>
const elemT*
device_readable_ptr(const Array<num_dimensions, elemT>& arr, CuVec<elemT>& tmp)
{
  if (onGPU(arr))
    return &*arr.begin_all();
  array_to_device(tmp, arr);
  return tmp.data();
}

//! Pointer a kernel can write to: the array's own pointer if on the GPU, otherwise \a tmp.
//! If \a copy_in is true, \a tmp is first filled with the array's contents.
//! If the array is not on the GPU, the caller must copy the result back afterwards.
template <int num_dimensions, typename elemT>
elemT*
device_writable_ptr(Array<num_dimensions, elemT>& arr, CuVec<elemT>& tmp, const bool copy_in)
{
  if (onGPU(arr))
    return &*arr.begin_all();
  if (copy_in)
    array_to_device(tmp, arr);
  else
    tmp.resize(arr.size_all());
  return tmp.data();
}
} // namespace detail
#endif

#ifdef __CUDACC__

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
