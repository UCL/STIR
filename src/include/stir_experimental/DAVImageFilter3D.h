//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file

  \brief Implementation of stir::DAVImageFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans
  
*/

#ifndef __stir_DAVImageFilter3D_H__
#define __stir_DAVImageFilter3D_H__


#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir_experimental/DAVArrayFilter3D.h"

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
