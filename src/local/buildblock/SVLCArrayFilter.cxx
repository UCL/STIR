#if 0

#include "tomo/SVLCArrayFilter.h"
#include "Coordinate3D.h"
// remove 
#include <iostream>
#include <fstream>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::streampos;
using std::cerr;
using std::endl;
using std::sort;
using std::min_element;
#endif


START_NAMESPACE_TOMO

template <typename elemT>
SVLCArrayFilter<elemT>::SVLCArrayFilter(const Coordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius[3];
  mask_radius_y = mask_radius[2];
  mask_radius_z = mask_radius[1];
}

void
template <typename elemT>
SVLCArrayFilter<elemT>::operator() (Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  elemT out_elem = 0;
  elemT mean =0;
  elemT variance =0;

  for (int z=in_array.get_min_index()+mask_radius_z;z<= in_array.get_max_index()-mask_radius_z;z++)
   for (int y=in_array[z].get_min_index()+mask_radius_y;y <= in_array[z].get_max_index()-mask_radius_y;y++)
     for (int x=in_array[z][y].get_min_index()+mask_radius_x;x <= in_array[z][y].get_max_index()-mask_radius_x;x++)       
     {
      find_lower_and_upper(lower,upper,in_array,Coordinate3D<int>& (z,y,x),beta);
      project_operator(out_elem,lower,upper,in_array,Coordinate3D<int>(z,y,x);
      out_array[z][y][x] = out_elem;  
     }
}

void 
template <typename elemT>
SVLCArrayFilter<elemT>::find_mean_and_variance(elemT& mean, elemT& variance, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const
{
  mean = 0;
  variance =0;
  int denom =(2*mask_radius_x)*(2*mask_radius_y)*(2*mask_radius_z);
  
  for (int z=-mask_radius_z; z <= mask_radius_z; z++)
    for (int y=-mask_radius_y; y <= mask_radius_y; y++)
      for (int x=-mask_radius_x; x<= mask_radius_x; x++)
      {
	mean += in_array[c_pixel[1]+z][c_pixel[2]+y][c_pixel[3]+x];	
      }
      mean /= denom;
      
      for (int z=-mask_radius_z; z <= mask_radius_z; z++)
	for (int y=-mask_radius_y; y <= mask_radius_y; y++)
	  for (int x=-mask_radius_x; x<= mask_radius_x; x++)
	  {
	    variance += square[in_array[c_pixel[1]][c_pixel[2]][c_pixel[3]]-mean];
	  }    
	  
}

void 
template <typename elemT>
SVLCArrayFilter<elemT>::find_lower_and_upper(elemT& lower, elemT& upper, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel, const float beta) const
{
  // initialize to 0 to prevent from warning
  elemT mean= 0;
  elemT variance= 0;
  find_median_and_variance(mean,variance,in_array,c_pixel);

  lower = max(mean - beta* variance,0);
  upper = mean + beta*variance;
}


END_NAMESPACE_TOMO

#endif