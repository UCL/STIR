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

#ifndef __tomo_Shape_EllipsoidalCylinder_h__
#define __tomo_Shape_EllipsoidalCylinder_h__

#include "tomo/RegisteredParsingObject.h"
#include "local/tomo/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_TOMO

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

END_NAMESPACE_TOMO

#include "local/tomo/Shape/EllipsoidalCylinder.inl"

#endif
