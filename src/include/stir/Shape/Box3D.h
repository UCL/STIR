//
//
/*
    Copyright (C) 2005- 2008, Hammersmith Imanet Ltd
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
  
  \brief Declaration of class stir::Box3D
  
  \par Description
  A point with coordinates \a coord is inside the shape if for
  x,y,z given by <code>Shape3DWithOrientation::transform_to_original_coords(coord)</code>
  \f[
  abs(x), abs(y), abs(z) <= length_x/2, length_y/2, length_z/2
\f]
  
  \par Parameters
  \verbatim
  Box3D Parameters:=
     length-x (in mm):= <float>
     length-y (in mm):= <float>
     length-z (in mm):= <float>
     ; any parameters of Shape3DWithOrientation
     End:=
  \endverbatim
  \author C. Ross Schmidtlein
*/

#ifndef __stir_Shape_Box3D_h__
#define __stir_Shape_Box3D_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Shape
  \brief Three-dimensional box
  
  box with the dimensions 
   (length_x,length_y,length_z), where length_x assumed to be
  in x direction, length_y in y direction, and length_z in z-direction,
   before any rotation with Euler angles.
*/
class Box3D: 
public RegisteredParsingObject<Box3D, Shape3D, Shape3DWithOrientation>
{
 public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 
  
  Box3D();
#if 0
  Box3D( const float length_x,
	 const float length_y,
	 const float length_z,
	 const CartesianCoordinate3D<float>& centre,
	 const float alpha,
	 const float beta,
	 const float gamma);
#endif

  Box3D( const float length_x, 
	 const float length_y,
	 const float length_z,
	 const CartesianCoordinate3D<float>& centre,
	 const Array<2,float>& direction_vectors = diagonal_matrix(3,1.F));

  float get_geometric_volume() const;
  float get_geometric_area() const;
  
  bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const;
  
  Shape3D* clone() const;

  //! Compare boxes
  /*! Uses a tolerance determined by the smallest dimension of the object divided by 1000.*/
  bool
    operator==(const Box3D&) const;

  virtual bool
    operator==(const Shape3D& shape) const;
  
 protected:
  //! Length in x-direction if the shape is not rotated
  float length_x;
  //! Length in y-direction if the shape is not rotated
  float length_y;
  //! Length in z-direction if the shape is not rotated
  float length_z;
 private:
  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
  
};

END_NAMESPACE_STIR

#endif
