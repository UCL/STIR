//
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans

   Warning: 
   At the moment it is essential to have 
   mask_radius_x = mask_radius_y =mask_radius_z = odd.

*/
/*
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_DAVArrayFilter3D_H__
#define __stir_DAVArrayFilter3D_H__

#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"


START_NAMESPACE_STIR

template <typename coordT> class Coordinate3D;


template <typename elemT>
class DAVArrayFilter3D: public ArrayFunctionObject_2ArgumentImplementation<3,elemT>
{
public:
  DAVArrayFilter3D (const Coordinate3D<int>& mask_radius = Coordinate3D<int>());    
  bool is_trivial() const;

private:
   int mask_radius_x;
   int mask_radius_y;
   int mask_radius_z;

  virtual void do_it (Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const;

  void extract_neighbours_and_average(elemT& out_elem, const Array<3,elemT>& in_array,const Coordinate3D<int>& c_pixel) const;

};

END_NAMESPACE_STIR

#endif



