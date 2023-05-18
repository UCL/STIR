//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class stir::DiscretisedShape3D

  \author Kris Thielemans
*/
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/round.h"
#include "stir/zoom.h"
#include "stir/is_null_ptr.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/IO/read_from_file.h"
START_NAMESPACE_STIR

void
DiscretisedShape3D::
set_origin(const CartesianCoordinate3D<float>& new_origin)
{ 
  assert(this->get_origin() == density_sptr->get_origin());
  Shape3D::set_origin(new_origin);
  density_sptr->set_origin(new_origin);
}

// TODO check code
float 
DiscretisedShape3D::
get_voxel_weight(
                 const CartesianCoordinate3D<float>& voxel_centre,
                 const CartesianCoordinate3D<float>& voxel_size, 
                 const CartesianCoordinate3D<int>& /* num_samples*/) const
{
  assert(this->get_origin() == density_sptr->get_origin());

  assert(voxel_size == image().get_voxel_size());
  const CartesianCoordinate3D<float> r = 
    this->density_sptr->get_index_coordinates_for_physical_coordinates(voxel_centre);

  const int x = round(r.x());
  const int y = round(r.y());
  const int z = round(r.z());
  // check that r points to the middle of a voxel
  assert(fabs(x-r.x())<=1E-6);
  assert(fabs(y-r.y())<=1E-6);
  assert(fabs(z-r.z())<=1E-6);
  if (z <= image().get_max_z() && z >= image().get_min_z() && y <= image().get_max_y() && y >= image().get_min_y()
      && x <= image().get_max_x() && x >= image().get_min_x())
    {
      if (this->_label_index >= 0)
        return (round(image()[z][y][x]) == this->_label_index) ? 1.F : 0.F;
      else
        return image()[z][y][x];
    }
  else
    return 0.F;
}

bool 
DiscretisedShape3D::
is_inside_shape(const CartesianCoordinate3D<float>& coord) const
{
  assert(this->get_origin() == density_sptr->get_origin());
  return 
    get_voxel_weight(coord, 
                     image().get_voxel_size(), 
                     CartesianCoordinate3D<int>(1,1,1)) > 0;
}

void 
DiscretisedShape3D::
construct_volume(VoxelsOnCartesianGrid<float> &new_image, const CartesianCoordinate3D<int>& num_samples) const
{
  if (this->_label_index >= 0)
    {
      std::string explanation;
      if (!this->image().has_same_characteristics(new_image, explanation))
        {
          error("DiscretisedShape3D:: construct_volume needs images with the same characteristics when using labels,"
                "but they did not match:\n"
                + explanation);
        }
      new_image = this->image();
      for (auto iter = new_image.begin_all(); iter != new_image.end_all(); ++iter)
        *iter = (round(*iter) == this->_label_index) ? 1.F : 0.F;
    }
  else
    zoom_image(new_image, this->image(), ZoomOptions::preserve_values);
}

Shape3D* 
DiscretisedShape3D::
clone() const
{
  assert(this->get_origin() == density_sptr->get_origin());

  return new DiscretisedShape3D(*this);
}

DiscretisedDensity<3,float>& 
DiscretisedShape3D::
get_discretised_density()
{
  return *density_sptr;
}

const DiscretisedDensity<3,float>& 
DiscretisedShape3D::
get_discretised_density() const
{
  return *density_sptr;
}

int
DiscretisedShape3D::get_label_index()  const
{
  return this->_label_index;
}

void
DiscretisedShape3D::set_label_index(int label)
{
  this->_label_index = label;
}

DiscretisedShape3D::
DiscretisedShape3D()
{
  set_defaults();
}

DiscretisedShape3D::
DiscretisedShape3D(const VoxelsOnCartesianGrid<float>& image_v)
   : _label_index(-1), density_sptr(image_v.clone())
{
  this->set_origin(image_v.get_origin());
  this->filename = "FROM MEMORY";
}
  


DiscretisedShape3D::
DiscretisedShape3D(const shared_ptr<const DiscretisedDensity<3,float> >& density_sptr_v)
    : _label_index(-1), density_sptr(density_sptr_v->clone())
{
  if(dynamic_cast<const VoxelsOnCartesianGrid<float> *>(density_sptr.get()) == NULL)
  {
    error("DiscretisedShape3D can currently only handle images of type VoxelsOnCartesianGrid.\n"); 
  }
  this->set_origin(density_sptr_v->get_origin());
  this->filename = "FROM MEMORY";
}


void 
DiscretisedShape3D::initialise_keymap()
{
  Shape3D::initialise_keymap();
  parser.add_start_key("Discretised Shape3D Parameters");
  parser.add_key("input filename", &filename);
  parser.add_key("label_index", &this->_label_index);
  parser.add_stop_key("END");
}



void
DiscretisedShape3D::set_defaults()
{  
  Shape3D::set_defaults();
  this->density_sptr.reset();
  this->_label_index = -1;
}

bool
DiscretisedShape3D::
post_processing()
{
  if (Shape3D::post_processing()==true)
    return true;

  density_sptr = read_from_file<DiscretisedDensity<3,float> >(filename);
  if (!is_null_ptr(density_sptr))
    {
      if (this->get_origin() != density_sptr->get_origin())
	{
	  warning("DiscretisedShape3D: Shape3D::origin and image origin are inconsistent. Using origin from image\n");
	  this->set_origin(density_sptr->get_origin());
	}
    }
  return is_null_ptr(density_sptr);
}

const char * const 
DiscretisedShape3D::registered_name = "Discretised Shape3D";

END_NAMESPACE_STIR
