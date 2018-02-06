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
  \ingroup ImageProcessor
  \brief Implementations for class MinimalImageFilter3D

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#ifndef __stir_MinimalImageFilter3D_H__
#define __stir_MinimalImageFilter3D_H__


#include "stir/DataProcessor.h"
#include "stir/MinimalArrayFilter3D.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename coordT> class CartesianCoordinate3D;



/*!
  \ingroup ImageProcessor
  \brief A class in the ImageProcessor hierarchy that implements minimal 
  filtering.
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.
 */
template <typename elemT>
class MinimalImageFilter3D:
  public 
      RegisteredParsingObject<
	      MinimalImageFilter3D<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
{
 private:
  typedef
    RegisteredParsingObject<
	      MinimalImageFilter3D<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
    base_type;
public:
  static const char * const registered_name; 

  MinimalImageFilter3D();

  MinimalImageFilter3D(const CartesianCoordinate3D<int>& mask_radius);  
 
private:
  MinimalArrayFilter3D<elemT> minimal_filter;
  int mask_radius_x;
  int mask_radius_y;
  int mask_radius_z;


  virtual void set_defaults();
  virtual void initialise_keymap();

  Succeeded virtual_set_up (const DiscretisedDensity< 3,elemT>& density);
  void virtual_apply(DiscretisedDensity<3,elemT>& density, const DiscretisedDensity<3,elemT>& in_density) const; 
  void virtual_apply(DiscretisedDensity<3,elemT>& density) const; 
};


END_NAMESPACE_STIR


#endif  // MinimalImageFilter3D
