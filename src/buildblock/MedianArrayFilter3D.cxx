//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Implementations for class stir::MedianArrayFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
/* History:
   SM first version
   KT 19/03/2002 
   change handling of edges (they were not filtered before)
   correct bug in case output and input array size were not the same
*/
#include "stir/MedianArrayFilter3D.h"
#include "stir/Coordinate3D.h"

#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::nth_element;
#endif

START_NAMESPACE_STIR


template <typename elemT>
MedianArrayFilter3D<elemT>::MedianArrayFilter3D(const Coordinate3D<int>& mask_radius)
{
  this->mask_radius_x = mask_radius[3];
  this->mask_radius_y = mask_radius[2];
  this->mask_radius_z = mask_radius[1];

 /* assert(mask_radius_x>0);
  assert(mask_radius_x%2 == 1);
  assert(mask_radius_y>0);
  assert(mask_radius_y%2 == 1);
  assert(mask_radius_z>0);
  assert(mask_radius_z%2 == 1);*/
}

template <typename elemT>
MedianArrayFilter3D<elemT>::MedianArrayFilter3D()
{
  this->mask_radius_x = 0;
  this->mask_radius_y = 0;
  this->mask_radius_z = 0;
}


template <typename elemT>
int
MedianArrayFilter3D<elemT>::
extract_neighbours(Array<1,elemT>& neigbours,const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const
{  
  int index=0;
  
  for (int zi =-mask_radius_z;zi<= mask_radius_z;++zi)
    {
      const int z = c_pixel[1]+zi;
      if (z<in_array.get_min_index() || z> in_array.get_max_index())
	continue;
      for (int yi =-mask_radius_y;yi<= mask_radius_y;++yi)
	{
	  const int y = c_pixel[2]+yi;
	  if (y<in_array[z].get_min_index() || y> in_array[z].get_max_index())
	    continue;
	  for( int xi=-mask_radius_x;xi<= mask_radius_x;++xi)
	    {
	      const int x = c_pixel[3]+xi;
	      if (x<in_array[z][y].get_min_index() || x> in_array[z][y].get_max_index())
		continue;
	      neigbours[index++] = in_array[z][y][x];
	    }
	}
    }
  return index;
}

template <typename elemT>
void
MedianArrayFilter3D<elemT>::
do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  assert(out_array.get_index_range() == in_array.get_index_range());

  Array<1,elemT> neighbours (0,(2*mask_radius_x+1)*(2*mask_radius_y+1)*(2*mask_radius_z+1)-1);

  for (int z=out_array.get_min_index();z<= out_array.get_max_index();++z)
   for (int y=out_array[z].get_min_index();y <= out_array[z].get_max_index();++y)
     for (int x=out_array[z][y].get_min_index();x <= out_array[z][y].get_max_index();++x)
     {
       const int num_neighbours =
	 extract_neighbours(neighbours,in_array,Coordinate3D<int>(z,y,x));
       if (num_neighbours==0)
         continue;
       nth_element(neighbours.begin(), neighbours.begin()+num_neighbours/2, neighbours.end());
       if (num_neighbours%2==1)
	 out_array[z][y][x] = neighbours[num_neighbours/2]; 
       else
	 out_array[z][y][x] = (neighbours[num_neighbours/2]+
			       neighbours[num_neighbours/2 - 1])/2; 
     } 
}


template <typename elemT>
bool
MedianArrayFilter3D<elemT>::
is_trivial() const
{
  if (mask_radius_x!=1 &&mask_radius_y !=1 &&mask_radius_z!=1)
    return true;
  else
    return false;

}


// instantiation
template class MedianArrayFilter3D<float>;

END_NAMESPACE_STIR

