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
#include <vector>

START_NAMESPACE_STIR

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

END_NAMESPACE_STIR

#endif
