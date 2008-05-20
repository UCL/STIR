/*!
  \file
  \ingroup Shape
  
  \brief Declaration of class Box3D
  
  \par Description
  A point with coordinates \a coord is inside the shape if for
  \f$x = coord - origin\f$:
  \f$y = coord - origin\f$:
  \f$z = coord - origin\f$:
  \f[
  abs(x), abs(y), abs(z) <= length_x/2, length_y/2, length_z/2
\f]
  where \f$dir_x, dir_y, dir_z\f$ are described in the documentation for class
  Shape3DWithOrientation.
  
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
  Box3D( const float length_x,
	 const float length_y,
	 const float length_z,
	 const CartesianCoordinate3D<float>& centre,
	 const float alpha,
	 const float beta,
	 const float gamma);
  Box3D( const float length_x, 
	 const float length_y,
	 const float length_z,
	 const CartesianCoordinate3D<float>& centre,
	 const CartesianCoordinate3D<float>& dir_x,
	 const CartesianCoordinate3D<float>& dir_y,
	 const CartesianCoordinate3D<float>& dir_z);
  void scale(const CartesianCoordinate3D<float>& scale3D);
  float get_geometric_volume() const;
  float get_geometric_area() const;
  
  bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const;
  
  Shape3D* clone() const;
  
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
