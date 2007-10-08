//
// $Id$
//
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
/*!

  \file

  \brief Implementation of stir::DAVImageFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#ifndef __stir_DAVImageFilter3D_H__
#define __stir_DAVImageFilter3D_H__


#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "local/stir/DAVArrayFilter3D.h"

#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename coordT> class CartesianCoordinate3D;

template <typename elemT>
class DAVImageFilter3D:
  public 
      RegisteredParsingObject<
	      DAVImageFilter3D<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >

{
public:
  static const char * const registered_name; 

  DAVImageFilter3D();

  DAVImageFilter3D(const CartesianCoordinate3D<int>& mask_radius);  
 
private:
  DAVArrayFilter3D<elemT> dav_filter;
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


#endif  // DAVImageFilter3D
