//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class Ellipsoid

  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_Shape_Ellipsoid_h__
#define __stir_Shape_Elliposoid_h__


#include "stir/RegisteredParsingObject.h"
#include "local/stir/Shape/Shape3DWithOrientation.h"




START_NAMESPACE_STIR

/*!
  \ingroup Shape
  \brief Three-dimensional ellipsoid
  
   Ellipsoid with the dimensions 
  (radius_a,radius_b,radius_c), where radius_a assumed to be
   in x direction, radius_b in y direction,radius_c in z-direction,
   before any rotation with Euler angles.
*/
class Ellipsoid: 
   public RegisteredParsingObject<Ellipsoid, Shape3D, Shape3DWithOrientation>
{
public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 

 Ellipsoid();
 Ellipsoid( const float radius_a,
	    const float radius_b,
	    const float radius_c,
	    const CartesianCoordinate3D<float>& centre,
	    const float alpha,
	    const float beta,
	    const float gamma); 
 Ellipsoid(  const float radius_a, 
             const float radius_b,
	     const float radius_c,
	     const CartesianCoordinate3D<float>& centre,
             const CartesianCoordinate3D<float>& dir_x,
             const CartesianCoordinate3D<float>& dir_y,
             const CartesianCoordinate3D<float>& dir_z);
  float get_geometric_volume()const;
  bool is_inside_shape(const CartesianCoordinate3D<float>& index) const;

  Shape3D* clone() const;

private:
  float radius_a;
  float radius_b;
  float radius_c;

  virtual void set_defaults();  
  virtual void initialise_keymap();
  
};


END_NAMESPACE_STIR

#endif
