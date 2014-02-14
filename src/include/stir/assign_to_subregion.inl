
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
assign_to_subregion(Array<3,elemT>& input_array, 
                    const BasicCoordinate<3,int>& mask_location,
                    const BasicCoordinate<3,int>& half_mask_size,
                    const elemT& value)
{
  const int min_k_index = input_array.get_min_index();
  const int max_k_index = input_array.get_max_index();
  for ( int k = max(mask_location[1]-half_mask_size[1],min_k_index); k<= min(mask_location[1]+half_mask_size[1],max_k_index); ++k)
    {
      const int min_j_index = input_array[k].get_min_index();
      const int max_j_index = input_array[k].get_max_index();
      for ( int j = max(mask_location[2]-half_mask_size[2],min_j_index); j<= min(mask_location[2]+half_mask_size[2],max_j_index); ++j)
        {
          const int min_i_index = input_array[k][j].get_min_index();
          const int max_i_index = input_array[k][j].get_max_index();
          for ( int i = max(mask_location[3]-half_mask_size[3],min_i_index); i<= min(mask_location[3]+half_mask_size[3],max_i_index); ++i)
            input_array[k][j][i] = value;
        }
    } 
}        
    
END_NAMESPACE_STIR



