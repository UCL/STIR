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

  \brief Declaration of class stir::DiscretisedShape3D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
#ifndef __stir_Shape_DiscretisedShape3D_H__
#define __stir_Shape_DiscretisedShape3D_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3D.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*! \ingroup Shape
  \brief A class for shapes that have been discretised

  Currently only supports discretisation via VoxelsOnCartesianGrid.

  For DiscretisedShaped3D objects with smooth edges, voxel values
    will vary between 0 and 1. 

*/
class DiscretisedShape3D: 
  public RegisteredParsingObject<DiscretisedShape3D, Shape3D, Shape3D>
{
public:
  //! Name which will be used when parsing a Shape3D object
  static const char * const registered_name; 

  DiscretisedShape3D();

  //! Constructor that will copy the image to an internal member
  /*! The \c filename member is set to "FROM MEMORY" such that parameter_info() 
      returns somewhat useful info. This has a consequence that the object cannot
      be constructed from its own parameter_info(). This is in contrast with most
      other shapes.
  */
  DiscretisedShape3D(const VoxelsOnCartesianGrid<float>& image);

  //! Constructor that will copy the shared_ptr image
  /*! The \c filename member is set to "FROM MEMORY" such that parameter_info() 
      returns somewhat useful info. This has a consequence that the object cannot
      be constructed from its own parameter_info(). This is in contrast with most
      other shapes.

      \warning any modifications to the object that this shared_ptr points to
      (and hence any modifications to this shape) will have
      confusing consequences.
  */
  DiscretisedShape3D(const shared_ptr<DiscretisedDensity<3,float> >& density_ptr);

  //! Compare shapes
  /*! \todo currently not implemented (will call error() */
  virtual bool
    operator==(const Shape3D&) const
  { error("DiscretisedShape3D::operator== not implemented. Sorry"); return false;}

  //! set origin of the shape
  /*! \warning this will shift the origin of the object pointed to by \a density_ptr.
     This is dangerous if you used the constructor taking a shared_ptr argument.*/
  virtual void set_origin(const CartesianCoordinate3D<float>&);

#ifdef DOXYGEN_SKIP
  // following lines here are only read by doxygen. 
  // They include a comment to warn the user. 
  // In actual fact, the function does not need to be reimplemented as it uses set_origin()
  //! translate the object by shifting its origin
  /*! \warning this will shift the origin of the object pointed to by \a density_ptr as it uses set_origin().
      \warning This function is in fact not reimplemented from the
      base_type.
  */
  void translate(const CartesianCoordinate3D<float>& direction);
#endif

  //! Scale shape
  /*! \todo Not implemented (will call error()) */
  virtual void scale(const CartesianCoordinate3D<float>& scale3D) 
  { error ("TODO: DiscretisedShape3D::scale not implemented. Sorry.");}

  //! determine if a point is inside a non-zero voxel or not
  /*! 
    \warning For voxels at the edges, it is somewhat
    ill-defined if a point in the voxel is inside the shape. The current
    implementation will return true for every point in the voxel, 
    even if the voxel value is .001.
    In particular, this means that this definition of is_inside_shape()
    cannot be used to find the voxel_weight. So, we have to redefine 
    get_voxel_weight() in the present class.
    */
  bool is_inside_shape(const CartesianCoordinate3D<float>& index) const;

  //! get weight for a voxel centred around \a coord
  /*! 
    \warning Presently only works when \a coord is the centre of a voxel and
          \a voxel_size is identical to the image's voxel_size

    The argument \a num_samples is ignored.
  */
 virtual float get_voxel_weight(
   const CartesianCoordinate3D<float>& coord,
   const CartesianCoordinate3D<float>& voxel_size, 
   const CartesianCoordinate3D<int>& num_samples) const;

 //! Construct a new image (using zoom_image) from the underlying density
 /*! 
   If the images do not have the same characteristics, zoom_image is called for interpolation.
   The result is scaled such that mean ROI values remain the same (at least for ROIs which avoid edges).

   The argument \a num_samples is ignored.
  */
  void construct_volume(VoxelsOnCartesianGrid<float> &image, const CartesianCoordinate3D<int>& num_samples) const;
 //void construct_slice(PixelsOnCartesianGrid<float> &plane, const CartesianCoordinate3D<int>& num_samples) const;
 
 
 virtual Shape3D* clone() const;

 //! provide access to the underlying density
 DiscretisedDensity<3,float>& get_discretised_density();

 //! provide (const) access to the underlying density
 const DiscretisedDensity<3,float>& get_discretised_density() const;
  
private:
  shared_ptr<DiscretisedDensity<3,float> > density_ptr;
  
  inline const VoxelsOnCartesianGrid<float>& image() const;
  inline VoxelsOnCartesianGrid<float>& image();

  //! \name Parsing functions
  //@{
  virtual void set_defaults();  
  virtual void initialise_keymap();
  //! Checks validity of parameters
  /*! As currently there are 2 origin parameters (in Shape3D and
      DiscretisedDensity), this function checks for consistency.
      However, the origin in Shape3D will be ignored.
  */
  virtual bool post_processing();
  //@}
  string filename;
};


END_NAMESPACE_STIR

#include "stir/Shape/DiscretisedShape3D.inl"

#endif
