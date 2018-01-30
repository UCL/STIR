/*
    Copyright (C) 2000-2008, Hammersmith Imanet Ltd
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
  \ingroup Shape

  \brief Declaration of class stir::EllipsoidalCylinder
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author C. Ross Schmidtlein (Added overload functions to define sectors of a cylinder)
*/

#ifndef __stir_Shape_EllipsoidalCylinder_h__
#define __stir_Shape_EllipsoidalCylinder_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Shape
  \brief Three-dimensional ellipsoidal cylinder
  

  \par Description
  A point with coordinates \a coord is inside the shape if for
  \f$r = coord - origin\f$:
  \f[
  |r.dir_z|<L/2
  \f]
  and
  \f[
  {(r.dir_x)^2 \over R_x^2} + {(r.dir_y)^2 \over R_y^2} <= 1
  \f]
  and
  \f[
   \theta_1 <= atan2(r.dir_y,r.dir_x) <= \theta_2
  \f]

  where \f$dir_x, dir_y, dir_z\f$ are described in the documentation for class
  Shape3DWithOrientation.

  \par Parameters
   To specify an ellipsoidal cylinder with the dimensions 
  (radius_x,radius_y,length, theta_1, theta_2), where radius_x is
   in x direction, radius_y in y direction, length in z-direction,
   theta_1 and theta_2 are defined counter clockwise from the positive x-axis 
   about the z-axis (before any rotations), use:
  \verbatim
     Ellipsoidal Cylinder Parameters:=
     radius-x (in mm):= <float>
     radius-y (in mm):= <float>
     length-z (in mm):= <float>
     initial angle (in deg):= <float> ; (defaults to 0)
     final_angle (in deg):= <float>   ; (defaults to 360)
     ; any parameters of Shape3DWithOrientation
     End:=
  \endverbatim
*/
class EllipsoidalCylinder: 
   public RegisteredParsingObject<EllipsoidalCylinder, Shape3D, Shape3DWithOrientation>
{

public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 
  
  //! Default constructor (calls set_defaults())
  EllipsoidalCylinder();

  //! Constructor
  /*! \warning: note order of arguments */
  EllipsoidalCylinder(const float length_z, 
                      const float radius_y,
                      const float radius_x,
                      const CartesianCoordinate3D<float>& centre,
		      const Array<2,float>& direction_vectors = diagonal_matrix(3,1.F));
  
  //! Constructor
  /*! \warning: note order of arguments.
      \bug angles \a theta_1 and \a theta_2 are currently in degrees, while STIR conventions dictate radians.
  */
  EllipsoidalCylinder(const float length_z, 
                      const float radius_y,
                      const float radius_x,
	              const float theta_1,
	              const float theta_2,
                      const CartesianCoordinate3D<float>& centre,
		      const Array<2,float>& direction_vectors = diagonal_matrix(3,1.F));

  Shape3D* clone() const; 

  //! Compare cylinders
  /*! Uses a tolerance determined by the smallest dimension of the object divided by 1000.*/
  bool
    operator==(const EllipsoidalCylinder& cylinder) const;

  virtual bool
    operator==(const Shape3D& shape) const;

  //! get volume
  float get_geometric_volume() const;
#if 0
  //! Get approximate geometric area
  float get_geometric_area() const;
#endif

  bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const;
  
  inline float get_length() const
    { return length; }
  inline float get_radius_x() const
    { return radius_x; }
  inline float get_radius_y() const
    { return radius_y; }
  //TODOXXX add theta_1,2
  void set_length(const float);
  void set_radius_x(const float);
  void set_radius_y(const float);

protected:

  //! Length of the cylinder
  float length;
  //! Radius in x-direction if the shape is not rotated
  float radius_x;
  //! Radius in y-direction if the shape is not rotated
  float radius_y;
  //! initial theta if the shape is not rotated (in degrees)
    float theta_1;
  //! final theta if the shape is not rotated (in degrees)
    float theta_2;

  //! set defaults before parsing
  /*! sets radii and length to 0, theta_1=0, theta_2=360 and calls Shape3DWithOrientation::set_defaults() */
  virtual void set_defaults();  
  virtual void initialise_keymap();    
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
