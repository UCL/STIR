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

  \brief Declaration of class stir::Shape3DWithOrientation

  \todo document parsing parameters
  \par Parameters
  \verbatim
     ; any parameters of Shape3D

     ; parameters that enable to use non-default axes
     ; values below are give a rotation around y for 90 degrees (swapping x and z)
     ; Warning: this uses the STIR convention {z,y,x}
     direction vectors (in mm) := { {0,0,1}, {0,1,0}, {-1,0,0}}
     End:=
  \endverbatim

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_Shape_Shape3DWithOrientation__H__
#define __stir_Shape_Shape3DWithOrientation__H__

#include "stir/Shape/Shape3D.h"
#include "stir/numerics/MatrixFunction.h"

START_NAMESPACE_STIR
class Succeeded;

/*!
  \ingroup Shape
  \brief Class for shapes with orientation

  Orientation is specified by giving a 3x3 matrix specifying 3 direction vectors.
  Note that these vectors do not necessarily have to be orthogonal nor have unit-norm
  (in fact, scale() will rescale these direction vectors). Of course, they should not
  be parallel, but this is not checked.

  Functions like \c is_inside_shape(coord) should compute the coordinate to be used
  in the calculation as <code>matrix_multiply(direction_vectors, coord-origin)</code>,
  or best practice is to call transform_to_original_coords().

  \todo A previous release had Euler angle code. However, it is currently disabled as 
  there were bugs in it.
*/

class Shape3DWithOrientation: public Shape3D
{
  typedef Shape3D base_type;

public:

  bool operator==(const Shape3DWithOrientation& s) const;
  virtual void scale(const CartesianCoordinate3D<float>& scale3D);

  //! get direction vectors currently in use
  /*! Index offsets will always be 1 */
  const Array<2,float>& get_direction_vectors() const
    { return _directions; }
  //! set direction vectors
  /*!
    Any index offset will be accepted.

    Note that scaling the direction vectors is equivalent to a call to
    scale_around_origin()
  */
  Succeeded set_direction_vectors(const Array<2,float>&);

#if 0
  // TODO non-sensical after non-uniform scale
  float get_angle_alpha() const;
  float get_angle_beta()  const;
  float get_angle_gamma() const;
#endif

protected:

  //! default constructor (NO initialisation of values)
  Shape3DWithOrientation();
  
#if 0
  /
  Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
			 const float alpha,
			 const float beta,
			 const float gamma);
#endif  
  Shape3DWithOrientation(const CartesianCoordinate3D<float>& origin,
			 const Array<2,float>& directions = diagonal_matrix(3,1.F));
  
#if 0
  void
    set_directions_from_Euler_angles(
				     const float alpha,
				     const float beta,
				     const float gamma);
#endif

  //! gets the volume of the cell spanned by the direction vectors
  /*! this should be used when computing volumes etc */
  float get_volume_of_unit_cell() const;

  //! Transform a 'real-world' coordinate to the coordinate system used by the shape
  CartesianCoordinate3D<float>
    transform_to_shape_coords(const CartesianCoordinate3D<float>&) const;

  //! sets defaults for parsing
  /*! sets direction vectors to the normal unit vectors. */
  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
  virtual void set_key_values();

private:
#if 0
  // temporary variables to store values while parsing
  float alpha_in_degrees;
  float beta_in_degrees;
  float gamma_in_degrees;
#endif

  Array<2,float> _directions;

};


END_NAMESPACE_STIR

#endif
