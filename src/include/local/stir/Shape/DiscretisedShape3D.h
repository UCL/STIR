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
#ifndef __tomo_Shape_DiscretisedShape3D_H__
#define __tomo_Shape_DiscretisedShape3D_H__

#include "tomo/RegisteredParsingObject.h"
#include "local/tomo/Shape/Shape3D.h"
#include "shared_ptr.h"

START_NAMESPACE_TOMO

template <int num_dimensions, typename elemT> class DiscretisedDensity;

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
    For template images with smooth edges, this means that this definition
    cannot be used to find the voxel_weight. This is of course why we redefine 
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

  virtual void set_defaults();  
  virtual void initialise_keymap();
  virtual bool post_processing();
    
  string filename;
};


END_NAMESPACE_TOMO

#include "local/tomo/Shape/DiscretisedShape3D.inl"

#endif
