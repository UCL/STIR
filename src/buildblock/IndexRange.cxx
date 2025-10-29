//
//
/*!
  \file
  \ingroup Array
  \brief implementations for the stir::IndexRange class

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/IndexRange.h"
#include <algorithm>

START_NAMESPACE_STIR

template <int num_dimensions>
bool
IndexRange<num_dimensions>::get_regular_range(BasicCoordinate<num_dimensions, int>& min,
                                              BasicCoordinate<num_dimensions, int>& max) const
{
  // check if empty range
  if (base_type::begin() == base_type::end())
    {
      std::fill(min.begin(), min.end(), 0);
      std::fill(max.begin(), max.end(), -1);
      return true;
    }

  // if not a regular range, exit
  if (is_regular_range == regular_false)
    return false;

  typename base_type::const_iterator iter = base_type::begin();

  BasicCoordinate<num_dimensions - 1, int> lower_dim_min;
  BasicCoordinate<num_dimensions - 1, int> lower_dim_max;
  if (!iter->get_regular_range(lower_dim_min, lower_dim_max))
    return false;

  if (is_regular_range == regular_to_do)
    {
      // check if all lower dimensional ranges have same regular range
      BasicCoordinate<num_dimensions - 1, int> lower_dim_min_try;
      BasicCoordinate<num_dimensions - 1, int> lower_dim_max_try;

      for (++iter; iter != base_type::end(); ++iter)
        {
          if (!iter->get_regular_range(lower_dim_min_try, lower_dim_max_try))
            {
              is_regular_range = regular_false;
              return false;
            }
          if (!std::equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin())
              || !std::equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
            {
              is_regular_range = regular_false;
              return false;
            }
        }
      // yes, they do
      is_regular_range = regular_true;
    }

  min = join(base_type::get_min_index(), lower_dim_min);
  max = join(base_type::get_max_index(), lower_dim_max);

  return true;
}

/***************************************************
 instantiations
 ***************************************************/

template class IndexRange<2>;
template class IndexRange<3>;
template class IndexRange<4>;
END_NAMESPACE_STIR
