//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class Shape3DWithOrientation

  \todo document parsing parameters

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Shape_Shape3DWithOrientation__H__
#define __stir_Shape_Shape3DWithOrientation__H__

#include "stir/Shape/Shape3D.h"
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif

/*!
  \ingroup Shape
  \brief Class for shapes with orientation
   
*/
START_NAMESPACE_STIR

class Shape3DWithOrientation: public Shape3D
{


public:

  void translate(const CartesianCoordinate3D<float>& direction);
 // void scale(const CartesianCoordinate3D<float>& scale3D);

  CartesianCoordinate3D<float> get_dir_x() const;
  CartesianCoordinate3D<float> get_dir_y() const;
  CartesianCoordinate3D<float> get_dir_z() const;
  // TODO non-sensical after non-uniform scale
  float get_angle_alpha() const;
  float get_angle_beta()  const;
  float get_angle_gamma() const;

protected:

  Shape3DWithOrientation();
  
  Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
    const float alpha,
    const float beta,
    const float gamma);
  
  Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
    const CartesianCoordinate3D<float>& dir_x,
    const CartesianCoordinate3D<float>& dir_y,
    const CartesianCoordinate3D<float>& dir_z);
  
  CartesianCoordinate3D<float> dir_x;
  CartesianCoordinate3D<float> dir_y;
  CartesianCoordinate3D<float> dir_z;

  void
    set_directions_from_Euler_angles(
                                 const float alpha,
                                 const float beta,
                                 const float gamma);

  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
  virtual void set_key_values();

private:
  // temporary variables to store values while parsing
  float alpha_in_degrees;
  float beta_in_degrees;
  float gamma_in_degrees;

  
  vector<double> dir_x_vector;
  vector<double> dir_y_vector;
  vector<double> dir_z_vector;
};


END_NAMESPACE_STIR

#endif
