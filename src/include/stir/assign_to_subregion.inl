
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array
  \brief implementation of the stir::assign_to_subregion function

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

START_NAMESPACE_STIR

template <class elemT>
void
assign_to_subregion(Array<3, elemT>& input_array,
                    const BasicCoordinate<3, int>& mask_location,
                    const BasicCoordinate<3, int>& half_mask_size,
                    const elemT& value)
{
  const int min_k_index = input_array.get_min_index();
  const int max_k_index = input_array.get_max_index();
  for (int k = std::max(mask_location[1] - half_mask_size[1], min_k_index);
       k <= std::min(mask_location[1] + half_mask_size[1], max_k_index);
       ++k)
    {
      const int min_j_index = input_array[k].get_min_index();
      const int max_j_index = input_array[k].get_max_index();
      for (int j = std::max(mask_location[2] - half_mask_size[2], min_j_index);
           j <= std::min(mask_location[2] + half_mask_size[2], max_j_index);
           ++j)
        {
          const int min_i_index = input_array[k][j].get_min_index();
          const int max_i_index = input_array[k][j].get_max_index();
          for (int i = std::max(mask_location[3] - half_mask_size[3], min_i_index);
               i <= std::min(mask_location[3] + half_mask_size[3], max_i_index);
               ++i)
            input_array[k][j][i] = value;
        }
    }
}

END_NAMESPACE_STIR
