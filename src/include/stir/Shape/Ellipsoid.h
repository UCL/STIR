//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

  \brief Declaration of class stir::Ellipsoid
  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#ifndef __stir_Shape_Ellipsoid_h__
#define __stir_Shape_Elliposoid_h__


#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3DWithOrientation.h"

START_NAMESPACE_STIR

/*!
  \ingroup Shape
  \brief Three-dimensional ellipsoid
  

  \par Description
  A point with coordinates \a coord is inside the shape if for
  \f$r = coord - origin\f$:
  \f[
  {(r.dir_x)^2 \over R_y^2} + {(r.dir_y)^2 \over R_y^2}+ + {(r.dir_z)^2 \over R_z^2} <= 1
  \f]
  where \f$dir_x, dir_y, dir_z\f$ are described in the documentation for class
  stir::Shape3DWithOrientation.

  \par Parameters
  \verbatim
      Ellipsoid Parameters:=
     radius-x (in mm):= <float>
     radius-y (in mm):= <float>
     radius-z (in mm):= <float>
     ; any parameters of Shape3DWithOrientation
     End:=
  \endverbatim
*/
class Ellipsoid: 
   public RegisteredParsingObject<Ellipsoid, Shape3D, Shape3DWithOrientation>
{
public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 

 Ellipsoid();
 Ellipsoid(  const CartesianCoordinate3D<float>& radii, 
	     const CartesianCoordinate3D<float>& centre,
	     const Array<2,float>& direction_vectors = diagonal_matrix(3,1.F));
  //! get volume
  float get_geometric_volume() const;
#if 0
  //! Get approximate geometric area
  float get_geometric_area() const;
#endif

  bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const;

  Shape3D* clone() const;

  //! Compare cylinders
  /*! Uses a tolerance determined by the smallest dimension of the object divided by 1000.*/
  bool
    operator==(const Ellipsoid&) const;

  virtual bool
    operator==(const Shape3D& shape) const;

  inline float get_radius_x() const
    { return radii.x(); }
  inline float get_radius_y() const
    { return radii.y(); }
  inline float get_radius_z() const
    { return radii.z(); }
  inline CartesianCoordinate3D<float> get_radii() const
    { return radii; }
  void set_radii(const CartesianCoordinate3D<float>& new_radii);


protected:
  //! Radii in 3 directions (before using the direction vectors)
  CartesianCoordinate3D<float> radii;

  //! set defaults before parsing
  /*! sets radii to 0 and calls Shape3DWithOrientation::set_defaults() */
  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
  
};


END_NAMESPACE_STIR

#endif
