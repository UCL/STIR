/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class Box3D

  \author C. Ross Schmidtlein

*/
#include "stir/Shape/Box3D.h"

START_NAMESPACE_STIR

const char * const 
Box3D::registered_name = "Box3D";

void 
Box3D::initialise_keymap()
{
  parser.add_start_key("box parameters");
  parser.add_key("length-x (in mm)", &length_x);
  parser.add_key("length-y (in mm)", &length_y);
  parser.add_key("length-z (in mm)", &length_z);
  parser.add_stop_key("END");
  Shape3DWithOrientation::initialise_keymap();
}

void
Box3D::set_defaults()
{  
  Shape3DWithOrientation::set_defaults();
  length_x=0;
  length_y=0;
  length_z=0;
}

bool
Box3D::
post_processing()
{
  if (Shape3DWithOrientation::post_processing()==true)
    return true;

  if (length_x <= 0)
    {
      warning("length_x should be positive, but is %g\n", length_x);
      return true;
    }
  if (length_y <= 0)
    {
      warning("length_y should be positive, but is %g\n", length_y);
      return true;
    }
  if (length_z <= 0)
    {
      warning("length_z should be positive, but is %g\n", length_z);
      return true;
    }
  return false;
}

Box3D::Box3D()
{
  set_defaults(); 
}

Box3D::Box3D(const float length_xv, 
	     const float length_yv,
	     const float length_zv,
	     const CartesianCoordinate3D<float>& centre_v,
	     const CartesianCoordinate3D<float>& dir_xv,
	     const CartesianCoordinate3D<float>& dir_yv,
	     const CartesianCoordinate3D<float>& dir_zv) 
  ://Shape3DWithOrientation(centre_v,dir_xv,dir_yv,dir_zv), 
  length_x(length_xv),
  length_y(length_yv),
  length_z(length_zv)
{
  origin = centre_v;
  dir_x = dir_xv;
  dir_y = dir_yv;
  dir_z = dir_zv;
}  

Box3D::Box3D(const float length_xv,
	     const float length_yv,
	     const float length_zv,
	     const CartesianCoordinate3D<float>& centre_v,
	     const float alpha_v,
	     const float beta_v,
	     const float gamma_v) 
  ://Shape3DWithOrientation(centre_v,alpha_v,beta_v,gamma_v),
  length_x(length_xv),
  length_y(length_yv),
  length_z(length_zv)
{
  origin = centre_v;
  set_directions_from_Euler_angles(alpha_v, beta_v, gamma_v);
}


bool Box3D::is_inside_shape(const CartesianCoordinate3D<float>& index) const
{
  const CartesianCoordinate3D<float> r = index - origin;
  
  const float distance_along_x_axis=
      inner_product(r,dir_x);
  const float distance_along_y_axis=
    inner_product(r,dir_y);
  const float distance_along_z_axis=
    inner_product(r,dir_z);
  
  if (fabs(distance_along_x_axis)<length_x/2)
    { 
      if (fabs(distance_along_y_axis)<length_y/2)
	{
	  if (fabs(distance_along_z_axis)<length_z/2) 
	    return true;
	}
    }
  else return false;
  return false;
}

void 
Box3D::scale(const CartesianCoordinate3D<float>& scale3D)
{
  if (norm(dir_z - CartesianCoordinate3D<float>(1,0,0)) > 1E-5F ||
      norm(dir_y - CartesianCoordinate3D<float>(0,1,0)) > 1E-5F ||	
      norm(dir_x - CartesianCoordinate3D<float>(0,0,1)) > 1E-5F)
    error("Box3D::scale cannot handle rotated case yet.\n");
  // TODO it's probably better to scale dir_x et al, but then other things might brake  (such as geometric_volume)

  origin *= scale3D;
  length_z *= scale3D.z();
  length_y *= scale3D.y();
  length_x *= scale3D.x();
}

float 
Box3D:: 
get_geometric_volume()const
{
   return static_cast<float>(length_x*length_y*length_z);
}


float 
Box3D:: 
get_geometric_area()const
{
  return static_cast<float>(2*(length_x*length_y+length_x
			       *length_z+length_y*length_z));
}

Shape3D* 
Box3D:: 
clone() const
{
  return static_cast<Shape3D *>(new Box3D(*this));
}

END_NAMESPACE_STIR
