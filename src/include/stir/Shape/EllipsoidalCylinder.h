//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class EllipsoidalCylinder

  \par Description
  A point with coordinates \a coord is inside the shape if for
  \f$r = coord - origin\f$:
  \f[
  |r.dir_z|<L/2
  \f]
  and
  \f[
  {(r.dir_x)^2 \over R_y^2} + {(r.dir_y)^2 \over R_y^2} <= 1
  \f]
  where \f$dir_x, dir_y, dir_z\f$ are described in the documentation for class
  Shape3DWithOrientation.

  \par Parameters
  \verbatim
      Ellipsoidal Cylinder Parameters:=
     radius-x (in mm):= <float>
     radius-y (in mm):= <float>
     length-z (in mm):= <float>
     ; any parameters of Shape3DWithOrientation
     End:=
  \endverbatim
  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Shape_EllipsoidalCylinder_h__
#define __stir_Shape_EllipsoidalCylinder_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

class EllipsoidalCylinder: 
   public RegisteredParsingObject<EllipsoidalCylinder, Shape3D, Shape3DWithOrientation>
{

public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 
  
  EllipsoidalCylinder();

  EllipsoidalCylinder(const float length, 
                      const float radius_x,
                      const float radius_y,
                      const CartesianCoordinate3D<float>& centre,
                      const CartesianCoordinate3D<float>& dir_x,
                      const CartesianCoordinate3D<float>& dir_y,
                      const CartesianCoordinate3D<float>& dir_z);
  
  
  EllipsoidalCylinder(const float length, 
                      const float radius_x,
                      const float radius_y,
                      const CartesianCoordinate3D<float>& centre,
                      const float alpha,
                      const float beta,
                      const float gamma); 
  Shape3D* clone() const; 
  

  //! Scale the cylinder
  /*! \todo This cannot handle a rotated cylinder yet. Instead, it will call error(). */
  void scale(const CartesianCoordinate3D<float>& scale3D);
  float get_geometric_volume() const;
  bool is_inside_shape(const CartesianCoordinate3D<float>&) const;
  
protected:

  //! Length of the cylinder
  float length;
  //! Radius in x-direction if the shape is not rotated
  float radius_x;
  //! Radius in y-direction if the shape is not rotated
  float radius_y;

private:
  virtual void set_defaults();  
  virtual void initialise_keymap();    
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
