//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class Ellipsoid

  \par Description
  A point with coordinates \a coord is inside the shape if for
  \f$r = coord - origin\f$:
  \f[
  {(r.dir_x)^2 \over R_y^2} + {(r.dir_y)^2 \over R_y^2}+ + {(r.dir_z)^2 \over R_z^2} <= 1
  \f]
  where \f$dir_x, dir_y, dir_z\f$ are described in the documentation for class
  Shape3DWithOrientation.

  \par Parameters
  \verbatim
      Ellipsoidal Cylinder Parameters:=
     radius-x (in mm):= <float>
     radius-y (in mm):= <float>
     radius-z (in mm):= <float>
     ; any parameters of Shape3DWithOrientation
     End:=
  \endverbatim
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_Shape_Ellipsoid_h__
#define __stir_Shape_Elliposoid_h__


#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Shape
  \brief Three-dimensional ellipsoid
  
   Ellipsoid with the dimensions 
  (radius_x,radius_y,radius_z), where radius_x assumed to be
   in x direction, radius_y in y direction,radius_z in z-direction,
   before any rotation with Euler angles.
*/
class Ellipsoid: 
   public RegisteredParsingObject<Ellipsoid, Shape3D, Shape3DWithOrientation>
{
public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 

 Ellipsoid();
 Ellipsoid( const float radius_x,
	    const float radius_y,
	    const float radius_z,
	    const CartesianCoordinate3D<float>& centre,
	    const float alpha,
	    const float beta,
	    const float gamma); 
 Ellipsoid(  const float radius_x, 
             const float radius_y,
	     const float radius_z,
	     const CartesianCoordinate3D<float>& centre,
             const CartesianCoordinate3D<float>& dir_x,
             const CartesianCoordinate3D<float>& dir_y,
             const CartesianCoordinate3D<float>& dir_z);
  float get_geometric_volume()const;
  bool is_inside_shape(const CartesianCoordinate3D<float>& index) const;

  Shape3D* clone() const;

protected:
  //! Radius in x-direction if the shape is not rotated
  float radius_x;
  //! Radius in y-direction if the shape is not rotated
  float radius_y;
  //! Radius in z-direction if the shape is not rotated
  float radius_z;
private:
  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
  
};


END_NAMESPACE_STIR

#endif
