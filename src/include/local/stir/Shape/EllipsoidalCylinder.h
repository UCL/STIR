//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class EllipsoidalCylinder

  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Shape_EllipsoidalCylinder_h__
#define __stir_Shape_EllipsoidalCylinder_h__

#include "stir/RegisteredParsingObject.h"
#include "local/stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

class EllipsoidalCylinder: 
   public RegisteredParsingObject<EllipsoidalCylinder, Shape3D, Shape3DWithOrientation>
{

public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 
  
  EllipsoidalCylinder();

  EllipsoidalCylinder(const float length, 
                      const float radius_a,
                      const float radius_b,
                      const CartesianCoordinate3D<float>& centre,
                      const CartesianCoordinate3D<float>& dir_x,
                      const CartesianCoordinate3D<float>& dir_y,
                      const CartesianCoordinate3D<float>& dir_z);
  
  
  EllipsoidalCylinder(const float length, 
                      const float radius_a,
                      const float radius_b,
                      const CartesianCoordinate3D<float>& centre,
                      const float alpha,
                      const float beta,
                      const float gamma); 
  
  inline void scale(const CartesianCoordinate3D<float>& scale3D);
  inline float get_geometric_volume() const;
  bool is_inside_shape(const CartesianCoordinate3D<float>& index) const;
  
  inline Shape3D* clone() const; 
  
private:

  float length;
  float radius_a;
  float radius_b;

  virtual void set_defaults();  
  virtual void initialise_keymap();    
};

END_NAMESPACE_STIR

#include "local/stir/Shape/EllipsoidalCylinder.inl"

#endif
