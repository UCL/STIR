//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
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
  \ingroup array

  \brief Declaration of stir:index_at_maximum() and stir::indices_at_maximum()

  \todo move implementations to .cxx
  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

/*! \ingroup array
  \brief Finds the index where the maximum occurs in a (1-dimensional) vector.

 If the maximum occurs more than once, the smallest index is returned.

 If the vector is empty, the function returns 0.

 \todo make iterator version, or something that works on std::vector

*/

template <class elemT>
int index_at_maximum(const VectorWithOffset<elemT>& v)
{
  if (v.size() == 0)
    return 0;

  int index_at_max=v.get_min_index();
  elemT max_value=v[index_at_max];
  for (int index=v.get_min_index(); index<=v.get_max_index(); ++index)
    {
      const elemT value = v[index];
      if (value>max_value)
	{
	  index_at_max=index;
	  max_value=value;
	}
    }
  return index_at_max;
}

/*! \ingroup array
  \brief Finds the first (3-dimensional) index where the maximum occurs 
  in a (3-dimensional) array.

  \todo generalise to arbitrary dimensions
  \todo implementation currently cycles through the data twice
*/
template<class elemT>                         
BasicCoordinate<3,int> 
indices_at_maximum(const Array<3,elemT>& input_array)
{
  const elemT current_maximum = input_array.find_max();
  BasicCoordinate<3,int>  max_location, min_index, max_index; 
  
  bool found=false;    
  min_index[1] = input_array.get_min_index();
  max_index[1] = input_array.get_max_index();
	for ( int k = min_index[1]; k<= max_index[1] && !found; ++k)
	{
	  min_index[2] = input_array[k].get_min_index();
	  max_index[2] = input_array[k].get_max_index();
	  for ( int j = min_index[2]; j<= max_index[2] && !found; ++j)
	  {
	    min_index[3] = input_array[k][j].get_min_index();
	    max_index[3] = input_array[k][j].get_max_index();
	    for ( int i = min_index[3]; i<= max_index[3] && !found; ++i)
	      {
		if (input_array[k][j][i] == current_maximum)
		   {
		     max_location[1] = k;
		     max_location[2] = j;
		     max_location[3] = i;
		   }
	      }
	  }
	}
  found = true;		
  return max_location;	
}                            

END_NAMESPACE_STIR
