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

  \brief Declaration of class stir::Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

#ifndef __stir_Shape_Shape3D_h__
#define __stir_Shape_Shape3D_h__

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

template <typename elemT> class VoxelsOnCartesianGrid;


/*!
  \ingroup Shape
  \brief The base class for all 3 dimensional shapes

  Shape3D objects are intended to represent geometrical object with
  sharp boundaries. So, a point is inside or shape, or it is not (i.e.
  no fuzzyness).

  The only derived class where this is relaxed is DiscretisedShape3D.
  However, this then needs some special treatment for some member 
  functions, and you have to be somewhat careful with that class.

  \todo This could/should be generalised to allow general fuzzy shapes.
  Probably the only thing to change is to let is_inside_shape() return
  a float (between 0 and 1). This would solve some issues with 
  DiscretisedDhape3D.

  \todo The restriction to the 3D case for this base class largely comes
  from the construct_volume() member (and the origin parsing members)

  \todo This base class really should have no origin member.
  For example, DiscretisedShape3D now has effectively two.
  Instead, we should have an additional class Shape3DWithOrigin.
  Easy to do.

  \par Parsing
  This base class defines the following keywords for parsing
  \verbatim
  ; specify origin as {z,y,x}
  origin (in mm):= <float> ;defaults to {0,0,0}
  \endverbatim
*/
class Shape3D :
   public RegisteredObject<Shape3D>,
   public ParsingObject
{
public:

  
  virtual ~Shape3D() {}
  
  //! Compare shapes
  /*!
      \par Implementation note
      
      This virtual function has to be implemented in each final class of the hierarchy.
      However, Shape3D::operator== has an implementation that checks equality of the
      origin (up-to a tolerance of .001). Derived classes can call this implementation.
      */
  virtual 
    inline bool operator==(const Shape3D&) const = 0;

  //! Compare shapes
  inline bool operator!=(const Shape3D&) const;

  /*! 
    \brief Determine (approximately) the intersection volume of a voxel with the shape.

    \param voxel_centre is a cartesian coordinate in 'absolute' coordinates,
    i.e. in mm and <b>not</b> relative to the \a origin member.
    \param voxel_size is the voxel size in mm.
    \param num_samples determines the number of samples to take in z,y,x
    direction.
    \return a value between 0 and 1 representing the fraction of the
    voxel inside the shape

    In the Shape3D implementation, this is simply done by calling
    is_inside_shape() at various points in the voxel, and returning
    the average value. Obviously, this will only approximate the 
    intersection volume for very large \a num_samples, or when the
    voxel is completely inside the shape.
  */
  virtual float get_voxel_weight(
    const CartesianCoordinate3D<float>& voxel_centre,
    const CartesianCoordinate3D<float>& voxel_size, 
    const CartesianCoordinate3D<int>& num_samples) const;
  
  
  //! determine if a point is inside the shape or not (up to floating point errors)
  /*! 
    \param coord is a cartesian coordinate in 'absolute' coordinates,
    i.e. in mm and <b>not</b> relative to the \a origin member.
  
    This is really only well defined for shapes with sharp boundaries. 
    \see DiscretisedShape3D::is_inside_shape for some discussion.
    \todo replace by floating point return value?
  */
  virtual bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const = 0;
  
  //! translate the whole shape by shifting its origin 
  /*! Uses set_origin().

    \see scale()
  */
  virtual void translate(const CartesianCoordinate3D<float>& direction);
  //! scale the whole shape 
  /*! 
  Scaling the shape also shifts the origin of the shape: 
  new_origin = old_origin * scale3D.
  This is necessary such that combined shapes keep their correct relative
  positions. This means that scaling and translating is non-commutative.
  \code
  shape1=shape;
  shape1.translate(offset); shape1.scale(scale);
  shape2=shape;
  shape2.scale(scale); shape2.translate(offset*scale);
  assert(shape1==shape2);
  \endcode
 */
  virtual void scale(const CartesianCoordinate3D<float>& scale3D) = 0;
  
  //! scale the whole shape, keeping the centre at the same place
  inline void scale_around_origin(const CartesianCoordinate3D<float>& scale3D);
  
  /*!
    \brief construct an image representation the shape in a discretised manner

    In principle, each voxel is sub-sampled to allow smoother edges.
    \warning Shapes have to be larger than the voxel size for sensible results.
    For efficiency reasons, the current implementation of this function
    does a first pass through the image where is_inside_shape() is called
    only for the centre of the voxels. After this, only edge voxels are
    resampled. So, if a shape lies between the centre of all voxels,
    it will not be sampled at all.
  \todo Get rid of restriction to allow only VoxelsOnCartesianGrid<float>
  (but that's rather hard)
  \todo Potentially this should fill a DiscretisedShape3D.
  */
  virtual void construct_volume(VoxelsOnCartesianGrid<float> &image, const CartesianCoordinate3D<int>& num_samples) const;
  //virtual void construct_slice(PixelsOnCartesianGrid<float> &plane, const CartesianCoordinate3D<int>& num_samples) const;

  //! Compute approximate volume
  /*! As this is not possible/easy for all shapes, the default implementation 
    returns a negative number. The user should check this to see if the returned 
    value makes sense.
  */
  virtual float get_geometric_volume() const;
#if 0
  //! Compute approximate geometric area
  /*! As this is not possible/easy for all shapes, the default implementation 
    returns a negative number. The user should check this to see if the returned 
    value makes sense.
  */
  virtual float get_geometric_area() const;
#endif
  
  //TODO get_bounding_box() const;

  //! get the origin of the shape-coordinate system
  inline CartesianCoordinate3D<float> get_origin() const;
  //! set the origin of the shape-coordinate system
  virtual void set_origin(const CartesianCoordinate3D<float>&);
  
  //! Allocate a new Shape3D object which is a copy of the current one.
  virtual Shape3D* clone() const = 0;
  

  // need to overload this to avoid ambiguity between Object::parameter_info and ParsingObject::parameter_info()
  virtual std::string parameter_info();

protected:
  inline Shape3D();
  inline explicit Shape3D(const CartesianCoordinate3D<float>& origin);

  //! \name Parsing functions
  //@{
  virtual void set_defaults();  
  virtual void initialise_keymap();
  //@}
private:
  //! origin of the shape
  CartesianCoordinate3D<float> origin;

};

END_NAMESPACE_STIR

#include "stir/Shape/Shape3D.inl"

#endif
