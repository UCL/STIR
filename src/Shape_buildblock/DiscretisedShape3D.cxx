//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class DiscretisedShape3D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/round.h"

START_NAMESPACE_STIR

void
DiscretisedShape3D::
translate(const CartesianCoordinate3D<float>& direction)
{ 
  origin += direction; 
  density_ptr->set_origin(origin);
}


// TODO check code
float 
DiscretisedShape3D::
get_voxel_weight(
                 const CartesianCoordinate3D<float>& coord,
                 const CartesianCoordinate3D<float>& voxel_size, 
                 const CartesianCoordinate3D<int>& num_samples) const
{
  assert(voxel_size == image().get_voxel_size());
  CartesianCoordinate3D<float> r = (coord - origin)/image().get_voxel_size();
  const int x = round(r.x());
  const int y = round(r.y());
  const int z = round(r.z());
  // check that r points to the middle of a voxel
  assert(fabs(x-r.x())<=1E-6);
  assert(fabs(y-r.y())<=1E-6);
  assert(fabs(z-r.z())<=1E-6);
  if (  z <= image().get_max_z() && z >= image().get_min_z() && 
    y <= image().get_max_y() && y >= image().get_min_y() && 
    x <= image().get_max_x() && x >= image().get_min_x())
    return image()[z][y][x];
  else
    return 0.F;
}

bool 
DiscretisedShape3D::
is_inside_shape(const CartesianCoordinate3D<float>& coord) const
{
  return 
    get_voxel_weight(coord, 
                     image().get_voxel_size(), 
                     CartesianCoordinate3D<int>(1,1,1)) > 0;
}

void 
DiscretisedShape3D::
construct_volume(VoxelsOnCartesianGrid<float> &new_image, const CartesianCoordinate3D<int>& num_samples) const
{
  // TODO
  if (origin != new_image.get_origin())
    error("DiscretisedShape3D::construct_volume: cannot handle shifting of origin yet\n");
  if (image().get_voxel_size() != new_image.get_voxel_size())
    error("DiscretisedShape3D::construct_volume: cannot handle different voxel sizes yet\n");  
  if (image().get_index_range() != new_image.get_index_range())
    error("DiscretisedShape3D::construct_volume: cannot handle different index ranges yet (i.e  the size of the 2 images is different)\n");  
  new_image = image();
}

Shape3D* 
DiscretisedShape3D::
clone() const
{
  return new DiscretisedShape3D(*this);
}


DiscretisedShape3D::
DiscretisedShape3D()
{
  set_defaults();
}

DiscretisedShape3D::
DiscretisedShape3D(const VoxelsOnCartesianGrid<float>& image_v)
   : density_ptr(image_v.clone())
{
  origin = image_v.get_origin();
}
  


DiscretisedShape3D::
DiscretisedShape3D(const shared_ptr<DiscretisedDensity<3,float> >& density_ptr_v)
   : density_ptr(density_ptr_v)
{
  if(dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_ptr.get()) == NULL)
  {
    error("DiscretisedShape3D can currently only handle images of type VoxelsOnCartesianGrid.\n"); 
  }
  origin = density_ptr_v->get_origin();
}


void 
DiscretisedShape3D::initialise_keymap()
{
  Shape3D::initialise_keymap();
  parser.add_start_key("Discretised Shape3D Parameters");
  parser.add_key("input filename", &filename);
  parser.add_stop_key("END");
}



void
DiscretisedShape3D::set_defaults()
{  
  Shape3D::set_defaults();
}

bool
DiscretisedShape3D::
post_processing()
{
  if (Shape3D::post_processing()==true)
    return true;

  density_ptr = DiscretisedDensity<3,float>::read_from_file(filename);
  return density_ptr == 0;
}

const char * const 
DiscretisedShape3D::registered_name = "Discretised Shape3D";

END_NAMESPACE_STIR
