//
//
/*!
  \file
  \ingroup Array
  \brief Implementations for class stir::GammaArrayFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/GammaArrayFilter2D.h"
#include <cmath>
#include <algorithm>

START_NAMESPACE_STIR

template <typename elemT>
GammaArrayFilter2D<elemT>::GammaArrayFilter2D()
{}

template <typename elemT>
void
GammaArrayFilter2D<elemT>::do_it(Array<3, elemT>& out_array, const Array<3, elemT>& in_array) const
{
  assert(out_array.get_index_range() == in_array.get_index_range());

  const int sx = out_array[0][0].get_length();
  const int sy = out_array[0].get_length();
  const int sa = out_array.get_length();
  const int min_x = in_array[0][0].get_min_index();
  const int min_y = in_array[0].get_min_index();
  const float targetAverage = 0.25;

  for (int ia = 0; ia < sa; ia++)
    {
      // Find min and max values in the current slice
      float min_val = INFINITY, max_val = -INFINITY;
      for (int i = 0; i < sx; i++)
        {
          for (int j = 0; j < sy; j++)
            {
              min_val = std::min(in_array[ia][min_x + i][min_y + j], min_val);
              max_val = std::max(in_array[ia][min_x + i][min_y + j], max_val);
            }
        }

      // Normalize the slice
      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          out_array[ia][min_x + i][min_y + j] = (in_array[ia][min_x + i][min_y + j] - min_val) / (max_val - min_val);

      // Calculate average pixel value
      float averagePixelValue = 0.0;
      int count = 0;
      for (int i = 0; i < sx; i++)
        {
          for (int j = 0; j < sy; j++)
            {
              if (std::abs(out_array[ia][min_x + i][min_y + j]) > 0.1)
                {
                  count++;
                  averagePixelValue += out_array[ia][min_x + i][min_y + j];
                }
            }
        }
      averagePixelValue /= count;

      // Apply gamma correction
      float gamma_val = 1.0;
      if (averagePixelValue > 0.0)
        gamma_val = std::log(targetAverage) / std::log(averagePixelValue);

      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          out_array[ia][min_x + i][min_y + j] = std::pow(out_array[ia][min_x + i][min_y + j], gamma_val);

      // Restore original scale
      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          out_array[ia][min_x + i][min_y + j] = out_array[ia][min_x + i][min_y + j] * (max_val - min_val) + min_val;
    }
}

template <typename elemT>
bool
GammaArrayFilter2D<elemT>::is_trivial() const
{
  return false;
}

// Explicit template instantiation
template class GammaArrayFilter2D<float>;

END_NAMESPACE_STIR
