//
// $Id$
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_DAVImageFilter3D_H__
#define __stir_DAVImageFilter3D_H__


#include "stir/ImageProcessor.h"
#include "local/stir/DAVArrayFilter3D.h"

#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename coordT> class CartesianCoordinate3D;

template <typename elemT>
class DAVImageFilter3D:
  public 
      RegisteredParsingObject<
	      DAVImageFilter3D<elemT>,
              ImageProcessor<3,elemT>
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
