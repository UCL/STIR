//
//
/*!

  \file
  \ingroup Array
  \brief Implementations for class stir::WienerArrayFilter2D

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/WienerArrayFilter2D.h"

#include <algorithm>

using std::nth_element;

START_NAMESPACE_STIR

template <typename elemT>
WienerArrayFilter2D<elemT>::WienerArrayFilter2D()
{}

template <typename elemT>
void
WienerArrayFilter2D<elemT>::do_it(Array<3, elemT>& out_array, const Array<3, elemT>& in_array) const
{
  assert(out_array.get_index_range() == in_array.get_index_range());

  const int sx = out_array[0][0].get_length();
  const int sy = out_array[0].get_length();
  const int sa = out_array.get_length();
  const int min_x = in_array[0][0].get_min_index();
  const int min_y = in_array[0].get_min_index();
  const int ws = 9;

  for (int ia = 0; ia < sa; ia++)
    {
      std::vector<std::vector<float>> localMean(sx, std::vector<float>(sy, 0.0f));
      std::vector<std::vector<float>> localVar(sx, std::vector<float>(sy, 0.0f));

      float noise = 0.;

      for (int i = 1; i < sx - 1; i++)
        {
          for (int j = 1; j < sy - 1; j++)
            {
              localMean[i][j] = 0;
              localVar[i][j] = 0;

              for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                  localMean[i][j] += in_array[ia][min_x + i + k][min_y + j + l];
              localMean[i][j] /= ws;

              for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                  localVar[i][j] += in_array[ia][min_x + i + k][min_y + j + l] * in_array[ia][min_x + i + k][min_y + j + l];
              localVar[i][j] = localVar[i][j] / ws - localMean[i][j] * localMean[i][j];

              noise += localVar[i][j];
            }
        }
      noise /= (sx * sy);

      for (int i = 1; i < sx - 1; i++)
        {
          for (int j = 1; j < sy - 1; j++)
            {
              out_array[ia][min_x + i][min_y + j] = (in_array[ia][min_x + i][min_y + j] - localMean[i][j])
                                                        / std::max(localVar[i][j], noise) * std::max(localVar[i][j] - noise, 0.f)
                                                    + localMean[i][j];
            }
        }
    }
}

template <typename elemT>
bool
WienerArrayFilter2D<elemT>::is_trivial() const
{
  return false;
}

// instantiation
template class WienerArrayFilter2D<float>;

END_NAMESPACE_STIR
