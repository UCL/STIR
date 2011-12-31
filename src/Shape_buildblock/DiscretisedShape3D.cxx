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

  \brief Non-inline implementations for class stir::DiscretisedShape3D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/round.h"
#include "stir/zoom.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
START_NAMESPACE_STIR

void
DiscretisedShape3D::
set_origin(const CartesianCoordinate3D<float>& new_origin)
{ 
  assert(this->get_origin() == density_ptr->get_origin());
  Shape3D::set_origin(new_origin);
  density_ptr->set_origin(new_origin);
}

// TODO check code
float 
DiscretisedShape3D::
get_voxel_weight(
                 const CartesianCoordinate3D<float>& voxel_centre,
                 const CartesianCoordinate3D<float>& voxel_size, 
                 const CartesianCoordinate3D<int>& num_samples) const
{
  assert(this->get_origin() == density_ptr->get_origin());

  assert(voxel_size == image().get_voxel_size());
  const CartesianCoordinate3D<float> r = 
    this->density_ptr->get_index_coordinates_for_physical_coordinates(voxel_centre);

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
  assert(this->get_origin() == density_ptr->get_origin());
  return 
    get_voxel_weight(coord, 
                     image().get_voxel_size(), 
                     CartesianCoordinate3D<int>(1,1,1)) > 0;
}

void 
DiscretisedShape3D::
construct_volume(VoxelsOnCartesianGrid<float> &new_image, const CartesianCoordinate3D<int>& num_samples) const
{
  zoom_image(new_image, this->image());
  const float factor =
    (image().get_voxel_size().x()*image().get_voxel_size().y()*image().get_voxel_size().z())/
    (new_image.get_voxel_size().x()*new_image.get_voxel_size().y()*new_image.get_voxel_size().z());
  if (fabs(factor-1)>.01)
    new_image *= factor;
}

Shape3D* 
DiscretisedShape3D::
clone() const
{
  assert(this->get_origin() == density_ptr->get_origin());

  return new DiscretisedShape3D(*this);
}

DiscretisedDensity<3,float>& 
DiscretisedShape3D::
get_discretised_density()
{
  return *density_ptr;
}

const DiscretisedDensity<3,float>& 
DiscretisedShape3D::
get_discretised_density() const
{
  return *density_ptr;
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
  this->set_origin(image_v.get_origin());
  this->filename = "FROM MEMORY";
}
  


DiscretisedShape3D::
DiscretisedShape3D(const shared_ptr<DiscretisedDensity<3,float> >& density_ptr_v)
   : density_ptr(density_ptr_v)
{
  if(dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_ptr.get()) == NULL)
  {
    error("DiscretisedShape3D can currently only handle images of type VoxelsOnCartesianGrid.\n"); 
  }
  this->set_origin(density_ptr_v->get_origin());
  this->filename = "FROM MEMORY";
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

  density_ptr = read_from_file<DiscretisedDensity<3,float> >(filename);
  if (!is_null_ptr(density_ptr))
    {
      if (this->get_origin() != density_ptr->get_origin())
	{
	  warning("DiscretisedShape3D: Shape3D::origin and image origin are inconsistent. Using origin from image\n");
	  this->set_origin(density_ptr->get_origin());
	}
    }
  return is_null_ptr(density_ptr);
}

const char * const 
DiscretisedShape3D::registered_name = "Discretised Shape3D";

END_NAMESPACE_STIR
