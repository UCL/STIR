#if 0
//
// %W%: %E%
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans

  \date %E%
  \version %I%
   Warning: 
   At the moment it is essential to have 
   mask_radius_x = mask_radius_y =mask_radius_z = odd.

*/


#ifndef __SVLCArrayFilter_H__
#define __SVLCArrayFilter_H__

#include "tomo/ArrayFilter.h"


START_NAMESPACE_TOMO

template <typename coordT> class Coordinate3D;


template <typename elemT>
class SVLCArrayFilter: public ArrayFilter<3,elemT>
{
public:
  SVLCArrayFilter (const Coordinate3D<int>& mask_radius = Coordinate3D<int>());    
  virtual inline void operator() (Array<3,elemT>& array) const;
  virtual void operator() (Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const;
  bool is_trivial() const;

private:
   int mask_radius_x;
   int mask_radius_y;
   int mask_radius_z;
   
   //float beta;

   void find_mean_and_variance(elemT& mean, elemT& variance, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const; 
   void find_lower_and_upper_bound(elemT& lower, elemT& upper, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel,const float beta) const; 
   void project_operator( elemT& elem_out, const elemT& lower, const elemT& upper,const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel);


};

template <typename elemT>
void 
SVLCArrayFilter<elemT>::
operator() (Array<3,elemT>& array) const
{
   Array<3,elemT> copy_array( array);
   (*this)(array, copy_array);
}

END_NAMESPACE_TOMO

#endif



#endif