//
//
/*!

  \file
  \ingroup buildblock
  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/DAVArrayFilter3D.h"
#include "stir/Coordinate3D.h"
#include "stir/Array.h"
// remove 
#include <iostream>
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::streampos;
using std::cerr;
using std::endl;
using std::sort;
using std::min_element;
#endif


START_NAMESPACE_STIR


template <typename elemT>
DAVArrayFilter3D<elemT>::DAVArrayFilter3D(const Coordinate3D<int>& mask_radius)
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
DAVArrayFilter3D<elemT>::
do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  assert(out_array.get_index_range() == in_array.get_index_range());

  // initialize out_elem to prevent warning
  elemT out_elem = 0;

  for (int z=in_array.get_min_index()+mask_radius_z;z<= in_array.get_max_index()-mask_radius_z;z++)
   for (int y=in_array[z].get_min_index()+mask_radius_y;y <= in_array[z].get_max_index()-mask_radius_y;y++)
     for (int x=in_array[z][y].get_min_index()+mask_radius_x;x <= in_array[z][y].get_max_index()-mask_radius_x;x++)       
   {
      extract_neighbours_and_average(out_elem,in_array,Coordinate3D<int>(z,y,x));
      out_array[z][y][x] = out_elem;       
     } 
}

template <typename elemT>
void
DAVArrayFilter3D<elemT>::
extract_neighbours_and_average(elemT& out_elem, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const
{
  
   Array<1,float> averaged_arrays3D (0,8); 
   Array<1,float> averaged_arrays_subtracted3D (0,8);
   
   averaged_arrays3D.fill(0);
   averaged_arrays_subtracted3D.fill(0);     

   // TODO - different sizes for different directions
  int denom = (2*mask_radius_x+1);

  for(int index=-mask_radius_x;index<=mask_radius_x;index++)
  {    
    // for each z-plane
    averaged_arrays3D[0] += in_array[c_pixel[1]][c_pixel[2]+index][c_pixel[3]];	 
    averaged_arrays3D[1] += in_array[c_pixel[1]][c_pixel[2]-index][c_pixel[3]+index];	   
    averaged_arrays3D[2] += in_array[c_pixel[1]][c_pixel[2]][c_pixel[3]+index];           
    averaged_arrays3D[3] += in_array[c_pixel[1]][c_pixel[2]+index][c_pixel[3]+index];
    // for axial direction
    if (mask_radius_z !=0)
    {
      averaged_arrays3D[4] += in_array[c_pixel[1]+index][c_pixel[2]][c_pixel[3]];  
      averaged_arrays3D[5] += in_array[c_pixel[1]-index][c_pixel[2]+index][c_pixel[3]];   
      averaged_arrays3D[6] += in_array[c_pixel[1]+index][c_pixel[2]+index][c_pixel[3]];
      // extra two diagonals
      averaged_arrays3D[7] += in_array[c_pixel[1]+index][c_pixel[2]+index][c_pixel[3]+index];           
      averaged_arrays3D[8] += in_array[c_pixel[1]+index][c_pixel[2]-index][c_pixel[3]+index];	  
    }
    else 
    {
      for (int i = 4;i<=8; i++)
	averaged_arrays3D[i] = 0;
    }
    
  }
  
  for (int i = 0;i<=8;i++)
    averaged_arrays3D[i] /= denom;
  
  averaged_arrays_subtracted3D = averaged_arrays3D - in_array[c_pixel[1]][c_pixel[2]][c_pixel[3]]; 
  
  for (int i = 0;i<=8;i++)
    averaged_arrays_subtracted3D[i] = fabs(averaged_arrays_subtracted3D[i]);  
/*  
  int counter = 0;
  int ind = 0;
  float min_smooth_3D = averaged_arrays_subtracted3D[0];

  while ( counter <=8)
  {
    if (min_smooth_3D >averaged_arrays_subtracted3D[counter])
    {
      min_smooth_3D = averaged_arrays_subtracted3D[counter];     
      ind = counter;
    }
    counter++;
  }
*/
  Array<1,float>::difference_type min_abs_diff_idx =
    min_element(averaged_arrays_subtracted3D.begin(), 
                averaged_arrays_subtracted3D.end()) -
    averaged_arrays_subtracted3D.begin();

  out_elem = *(averaged_arrays3D.begin() + min_abs_diff_idx); 
}


template <typename elemT>
bool
DAVArrayFilter3D<elemT>::
is_trivial() const
{
  if (mask_radius_x!=1 &&mask_radius_y !=1 &&mask_radius_z!=1)
    return true;
  else
    return false;

}


// instantiation
template DAVArrayFilter3D<float>;

END_NAMESPACE_STIR

