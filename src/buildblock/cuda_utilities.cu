//
//
/*
    Copyright (C) 2026 National Physical Laboratory
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup CUDA
  \brief some utilities for STIR and CUDA

  \author Kris Thielemans
  \author Matteo Neel Colombo
  \author Daniel Deidda
*/
#include "stir/cuda_utilities.h"
START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
void
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

//! copy an `Array` to pre-allocated CuVec
/*!
  \ingroup CUDA
*/
template <int num_dimensions, typename elemT>
void
array_to_device(CuVec<elemT>& dev_data, const Array<num_dimensions, elemT>& stir_array)
{
  dev_data.resize(stir_array.size_all());
  std::copy(stir_array.begin_all(), stir_array.end_all(), dev_data.begin());
}

//! copy CUDA pointer to `Array`
/*!
  \ingroup CUDA
  The third argument is ignored, as `cudaMemcpy` always syncs device and host.
*/
template <int num_dimensions, typename elemT>
void
array_to_host(Array<num_dimensions, elemT>& stir_array, const elemT* dev_data, bool sync)
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

//! copy CuVec to `Array`
/*!
  \ingroup CUDA
  If \a sync = \c true, the function will call `cudaDeviceSynchronize()` before copying.
*/
template <int num_dimensions, typename elemT>
void
array_to_host(Array<num_dimensions, elemT>& stir_array, const CuVec<elemT>& dev_data, bool sync)
{
  if (sync)
    cudaDeviceSynchronize();
  if (stir_array.size_all() != dev_data.size())
    error("array_to_host: size mismatch between CuVec and Array");
  std::copy(dev_data.begin(), dev_data.end(), stir_array.begin_all());
}

template void array_to_device<3, float>(float*, const Array<3, float>&);

template void array_to_host<3, float>(Array<3, float>&, const float*, bool);

template void array_to_device<3, float>(CuVec<float>&, const Array<3, float>&);

template void array_to_host<3, float>(Array<3, float>&, const CuVec<float>&, bool);

template void array_to_device<3, double>(double*, const Array<3, double>&);

template void array_to_host<3, double>(Array<3, double>&, const double*, bool);

template void array_to_device<3, double>(CuVec<double>&, const Array<3, double>&);

template void array_to_host<3, double>(Array<3, double>&, const CuVec<double>&, bool);

template void array_to_device<2, float>(float*, const Array<2, float>&);

template void array_to_host<2, float>(Array<2, float>&, const float*, bool);

template void array_to_device<2, float>(CuVec<float>&, const Array<2, float>&);

template void array_to_host<2, float>(Array<2, float>&, const CuVec<float>&, bool);

template void array_to_device<2, double>(double*, const Array<2, double>&);

template void array_to_host<2, double>(Array<2, double>&, const double*, bool);

template void array_to_device<2, double>(CuVec<double>&, const Array<2, double>&);

template void array_to_host<2, double>(Array<2, double>&, const CuVec<double>&, bool);

template void array_to_device<1, float>(float*, const Array<1, float>&);

template void array_to_host<1, float>(Array<1, float>&, const float*, bool);

template void array_to_device<1, float>(CuVec<float>&, const Array<1, float>&);

template void array_to_host<1, float>(Array<1, float>&, const CuVec<float>&, bool);

template void array_to_device<1, double>(double*, const Array<1, double>&);

template void array_to_host<1, double>(Array<1, double>&, const double*, bool);

template void array_to_device<1, double>(CuVec<double>&, const Array<1, double>&);

template void array_to_host<1, double>(Array<1, double>&, const CuVec<double>&, bool);

END_NAMESPACE_STIR
