//
// $Id$
//
/*!

  \file

  \brief Implementations for class MedianArrayFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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
  mask_radius_x = mask_radius[3];
  mask_radius_y = mask_radius[2];
  mask_radius_z = mask_radius[1];

 /* assert(mask_radius_x>0);
  assert(mask_radius_x%2 == 1);
  assert(mask_radius_y>0);
  assert(mask_radius_y%2 == 1);
  assert(mask_radius_z>0);
  assert(mask_radius_z%2 == 1);*/
}


template <typename elemT>
void
MedianArrayFilter3D<elemT>::
extract_neighbours(Array<1,elemT>& neigbours,const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const
{  
  int index=0;
  
  for (int zi =-mask_radius_z;zi<= mask_radius_z;zi++)
    for (int yi =-mask_radius_y;yi<= mask_radius_y;yi++)
      for( int xi=-mask_radius_x;xi<= mask_radius_x;xi++)	 
        neigbours[index++] = in_array[c_pixel[1]+zi][c_pixel[2]+yi][c_pixel[3]+xi];
}

template <typename elemT>
void
MedianArrayFilter3D<elemT>::
do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  assert(out_array.get_index_range() == in_array.get_index_range());

  Array<1,elemT> neighbours (0,(2*mask_radius_x+1)*(2*mask_radius_y+1)*(2*mask_radius_z+1)-1);

  for (int z=in_array.get_min_index()+mask_radius_z;z<= in_array.get_max_index()-mask_radius_z;z++)
   for (int y=in_array[z].get_min_index()+mask_radius_y;y <= in_array[z].get_max_index()-mask_radius_y;y++)
     for (int x=in_array[z][y].get_min_index()+mask_radius_x;x <= in_array[z][y].get_max_index()-mask_radius_x;x++)       
     {
      extract_neighbours(neighbours,in_array,Coordinate3D<int>(z,y,x));	 
      nth_element(neighbours.begin(), neighbours.begin()+neighbours.get_length()/2, neighbours.end());
      out_array[z][y][x] = neighbours[neighbours.get_length()/2]; 
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
template MedianArrayFilter3D<float>;

END_NAMESPACE_STIR

