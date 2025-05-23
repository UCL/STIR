//
/*
 Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
 Copyright (C) 2010 - 2013, King's College London
 This file is part of STIR.
 
 SPDX-License-Identifier: Apache-2.0
 
 See STIR/LICENSE.txt for details
 */
/*!
  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::MaximalImageFilter3D
  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/

#ifndef __stir_MaximalImageFilter3D_H__
#define __stir_MaximalImageFilter3D_H__


#include "stir/DataProcessor.h"
#include "stir/MaximalArrayFilter3D.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename coordT> class CartesianCoordinate3D;



/*!
  \ingroup ImageProcessor
  \brief A class in the ImageProcessor hierarchy that implements maximal 
  filtering.
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.
 */
template <typename elemT>
class MaximalImageFilter3D:
  public 
      RegisteredParsingObject<
	      MaximalImageFilter3D<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
{
 private:
  typedef
    RegisteredParsingObject<
	      MaximalImageFilter3D<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
    base_type;
public:
  static const char * const registered_name; 

  MaximalImageFilter3D();

  MaximalImageFilter3D(const CartesianCoordinate3D<int>& mask_radius);  
 
private:
  MaximalArrayFilter3D<elemT> maximal_filter;
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


#endif  // MaximalImageFilter3D
