//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

#ifndef __tomo_Shape_Shape3D_h__
#define __tomo_Shape_Shape3D_h__

#include "tomo/RegisteredObject.h"
#include "tomo/ParsingObject.h"
#include "CartesianCoordinate3D.h"
#include <string>


#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO

template <typename elemT> class VoxelsOnCartesianGrid;


/*!
  \ingroup Shape
  \brief The base class for all 3 dimensional shapes
*/
class Shape3D :
   public RegisteredObject<Shape3D>,
   public ParsingObject
{
public:

  
  virtual ~Shape3D() {}
  
  virtual float get_voxel_weight(
    const CartesianCoordinate3D<float>& index,
    const CartesianCoordinate3D<float>& voxel_size, 
    const CartesianCoordinate3D<int>& num_samples) const;
  
  
  //! determine if a point is inside the shape or not (up to floating point errors)
  /*! 
  This is really only well defined for shapes with sharp boundaries. 
  \see DiscretisedShape3D::is_inside_shape.
  */
  virtual bool is_inside_shape(const CartesianCoordinate3D<float>& index) const = 0;
  
  //! translate the whole shape (see scale)
  virtual void translate(const CartesianCoordinate3D<float>& direction) = 0;
  //! scale the whole shape 
  /*! 
  Scaling the shape also shifts the centre of the shape: 
  new_centre = old_centre * scale3D.
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
  virtual void scale(const CartesianCoordinate3D<float>& scale3D) 
  { error ("TODO: scale");}
  
  //! scale the whole shape, keeping the centre at the same place
  inline void scale_around_origin(const CartesianCoordinate3D<float>& scale3D);
  
  
  virtual void construct_volume(VoxelsOnCartesianGrid<float> &image, const CartesianCoordinate3D<int>& num_samples) const;
  //virtual void construct_slice(PixelsOnCartesianGrid<float> &plane, const CartesianCoordinate3D<int>& num_samples) const;
  //virtual float get_geometric_volume() const =0;
  
  //TODO get_bounding_box() const;
  //! get the centre of the shape
  inline CartesianCoordinate3D<float> get_origin() const;
  
  //! Allocate a new Shape3D object which is a copy of the current one.
  virtual Shape3D* clone() const = 0;
 
protected:
  inline Shape3D();
  inline Shape3D(const CartesianCoordinate3D<float>& origin);

  CartesianCoordinate3D<float> origin;

  virtual void set_defaults();  
  virtual void initialise_keymap();

};

END_NAMESPACE_TOMO

#include "Shape3D.inl"

#endif
