//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class DiscretisedShape3D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
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

  DiscretisedShape3D(const VoxelsOnCartesianGrid<float>& image);

  DiscretisedShape3D(const shared_ptr<DiscretisedDensity<3,float> >& density_ptr);

  //! translate the object by shifting its origin
  /*! \warning this will shift the origin of the object pointed to by \a density_ptr.*/
  void translate(const CartesianCoordinate3D<float>& direction);
  // TODO
 // void scale(const CartesianCoordinate3D<float>& scale3D);

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

  void construct_volume(VoxelsOnCartesianGrid<float> &image, const CartesianCoordinate3D<int>& num_samples) const;
 //void construct_slice(PixelsOnCartesianGrid<float> &plane, const CartesianCoordinate3D<int>& num_samples) const;
 
 
 virtual Shape3D* clone() const;
  
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
      DiscretisedDensity, this function checks for consistency).
  */
  virtual bool post_processing();
  //@}
  string filename;
};


END_NAMESPACE_STIR

#include "stir/Shape/DiscretisedShape3D.inl"

#endif
