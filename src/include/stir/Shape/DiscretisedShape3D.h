//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Shape

  \brief Declaration of class stir::DiscretisedShape3D

  \author Kris Thielemans
*/
#ifndef __stir_Shape_DiscretisedShape3D_H__
#define __stir_Shape_DiscretisedShape3D_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3D.h"
#include "stir/shared_ptr.h"
#include "stir/error.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;

/*! \ingroup Shape
  \brief A class for shapes that have been discretised as a volume

  Currently only supports discretisation via VoxelsOnCartesianGrid.

  This class supports 2 options:
  - a label-image with associated label index (an integer), suitable for multiple ROIs in a single file.
  - a "weight" image, with (potentially) smooth edges, where voxel values
    vary between 0 and 1.

  \par Parameters for parsing
  \verbatim
  Discretised Shape3D Parameters:=
  input filename := <filename>
  label index := -1 ; if less than 1 (default), we will use "weights"
  END:=
  \endverbatim
  where \a filename needs to specify a volume that can be read by STIR.
*/
class DiscretisedShape3D : public RegisteredParsingObject<DiscretisedShape3D, Shape3D, Shape3D>
{
public:
  //! Name which will be used when parsing a Shape3D object
  static const char* const registered_name;

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
  */
  DiscretisedShape3D(const shared_ptr<const DiscretisedDensity<3, float>>& density_sptr);

  //! Compare shapes
  /*! \todo currently not implemented (will call error() */
  bool operator==(const Shape3D&) const override
  {
    error("DiscretisedShape3D::operator== not implemented. Sorry");
    return false;
  }

  //! set origin of the shape
  void set_origin(const CartesianCoordinate3D<float>&) override;

  //! Scale shape
  /*! \todo Not implemented (will call error()) */
  void scale(const CartesianCoordinate3D<float>& scale3D) override
  {
    error("TODO: DiscretisedShape3D::scale not implemented. Sorry.");
  }

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
  bool is_inside_shape(const CartesianCoordinate3D<float>& index) const override;

  //! get weight for a voxel centred around \a coord
  /*!
    \warning Presently only works when \a coord is the centre of a voxel and
          \a voxel_size is identical to the image's voxel_size

    The argument \a num_samples is ignored.

    If get_label_index() >= 0, the weight will be 1 for those voxels whose value is equal to the label_index and zero otherwise.
    If get_label_index() < 0 (default), the weight will be the actual voxel value.
  */
  float get_voxel_weight(const CartesianCoordinate3D<float>& coord,
                         const CartesianCoordinate3D<float>& voxel_size,
                         const CartesianCoordinate3D<int>& num_samples) const override;

  //! Construct a new image from the underlying density
  /*!
    If get_label_index() >= 0, the imags need to have the same characteristics, but in the other case,
    zoom_image is called for interpolation.
    The result is then scaled such that mean ROI values remain the same (at least for ROIs which avoid edges).

    The argument \a num_samples is ignored.
   */
  void construct_volume(VoxelsOnCartesianGrid<float>& image, const CartesianCoordinate3D<int>& num_samples) const override;
  // void construct_slice(PixelsOnCartesianGrid<float> &plane, const CartesianCoordinate3D<int>& num_samples) const;

  Shape3D* clone() const override;

  //! provide access to the underlying density
  DiscretisedDensity<3, float>& get_discretised_density();

  //! provide (const) access to the underlying density
  const DiscretisedDensity<3, float>& get_discretised_density() const;

  //! Return label index
  int get_label_index() const;
  //! Set label index
  void set_label_index(int label_index);

private:
  int _label_index;
  shared_ptr<DiscretisedDensity<3, float>> density_sptr;

  inline const VoxelsOnCartesianGrid<float>& image() const;
  // inline VoxelsOnCartesianGrid<float>& image();

  //! \name Parsing functions
  //@{
  //! Sets defaults i.e. label index=-1 and reset density_sptr
  void set_defaults() override;
  void initialise_keymap() override;
  //! Checks validity of parameters
  /*! As currently there are 2 origin parameters (in Shape3D and
      DiscretisedDensity), this function checks for consistency.
      However, the origin in Shape3D will be ignored.
  */
  bool post_processing() override;
  //@}
  std::string filename;
};

END_NAMESPACE_STIR

#include "stir/Shape/DiscretisedShape3D.inl"

#endif
