//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class MedianArrayFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_MedianArrayFilter3D_H__
#define __Tomo_MedianArrayFilter3D_H__

#include "tomo/ArrayFunctionObject_2ArgumentImplementation.h"


START_NAMESPACE_TOMO

template <typename coordT> class Coordinate3D;

/*!
  \ingroup buildblock
  \brief Implements median filtering on 3D arrays.
  */
// TODO generalise to n-dimensions
template <typename elemT>
class MedianArrayFilter3D: public ArrayFunctionObject_2ArgumentImplementation<3,elemT>
{
public:
  MedianArrayFilter3D (const Coordinate3D<int>& mask_radius = Coordinate3D<int>());    
  bool is_trivial() const;
  
private:
  int mask_radius_x;
  int mask_radius_y;
  int mask_radius_z;
  
  virtual void do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const;

  void extract_neighbours(Array<1,elemT>&,const Array<3,elemT>& array, const Coordinate3D<int>&) const;

};

END_NAMESPACE_TOMO

#endif



