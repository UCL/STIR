//
//
/*
    Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
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
  \brief Applies the erosion filter (i.e. voxel=min(neighbours))

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#ifndef __stir_MinimalArrayFilter3D_H__
#define __stir_MinimalArrayFilter3D_H__

#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"


START_NAMESPACE_STIR

template <typename coordT> class Coordinate3D;

/*!
  \ingroup Array
  \brief Implements erosion on 3D arrays.

  The minimum value for a 1D array of 2n+1 elements is defined as the minimum element
  of the sorted array. 

  For 3D images, the current filter works by extracting all neigbours (given by 
  the mask) to a 1D array, and getting the minimal of that array.

  This implementation of the minimal filter handles edges by taking the minimum of 
  all available pixels. For instance, when a 3x3 mask is used, and the 
  pixel-to-be-filtered is at the left edge, there will be only 6 pixels in the 
  mask (instead of 9).

  \todo Currently, the mask is determined in terms of the mask radius (in pixels), where
  size = 2*radius+1. This could easily be relaxed.
  \todo generalise to n-dimensions
  */
template <typename elemT>
class MinimalArrayFilter3D: public ArrayFunctionObject_2ArgumentImplementation<3,elemT>
{
public:
  explicit MinimalArrayFilter3D (const Coordinate3D<int>& mask_radius);
  MinimalArrayFilter3D ();    
  bool is_trivial() const;
  
private:
  int mask_radius_x;
  int mask_radius_y;
  int mask_radius_z;
  
  virtual void do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const;

  //! extract all neighbours and put them in a 1D array
  /*! \return the number of neighbours within the image range
   */
  int extract_neighbours(Array<1,elemT>&,const Array<3,elemT>& array, const Coordinate3D<int>&) const;

};

END_NAMESPACE_STIR

#endif

