//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

  \brief Non-inline implementations for class stir::Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
*/
#include "stir/Shape/Shape3D.h"
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


// Check the sampled elements of the voxel

START_NAMESPACE_STIR
/*
Shape3D* Shape3D::read_from_file(const string& filename)
{
  // at the moment only this one
  return new 
    DiscretisedShape3D(DiscretisedDensity<3,float>::read_from_file(filename));
}
*/

void
Shape3D::
set_origin(const CartesianCoordinate3D<float>& new_origin)
{
  this->origin = new_origin;
}


void
Shape3D::
translate(const CartesianCoordinate3D<float>& direction)
{ 
  this->set_origin(this->get_origin() + direction); 
}

float 
Shape3D::
get_geometric_volume() const
{
  return -1.F;
}

#if 0
float 
Shape3D::
get_geometric_area() const
{
  return -1.F;
}
#endif

float 
Shape3D::
get_voxel_weight(
		 const CartesianCoordinate3D<float>& voxel_centre,
		 const CartesianCoordinate3D<float>& voxel_size,
		 const CartesianCoordinate3D<int>& num_samples) const
{ 
  int value=0;
  
  for (float zsmall = -float(num_samples.z()-1)/num_samples.z()/2.F;
       zsmall<=0.5F;
       zsmall+=1.F/num_samples.z())
  {
    for (float ysmall =-float(num_samples.y()-1)/num_samples.y()/2.F;
	 ysmall<=0.5F;
	 ysmall+=1.F/num_samples.y())
    {
      for(float xsmall=-float(num_samples.x()-1)/num_samples.x()/2.F;
	  xsmall<=0.5F;
	  xsmall+=1.F/num_samples.x())
      {
	{
	  const CartesianCoordinate3D<float> r(zsmall,ysmall,xsmall);
	  if(is_inside_shape(voxel_centre+r*voxel_size))
	  value += 1;	  	   
	}
      }
    }
    
  }
  return float(value)/(num_samples.z()*num_samples.y()*num_samples.x());
}

/* Construct the volume- use the convexity, e.g
   the inner voxels sampled with num_samples=1, only the outer
   voxels checked with the user defined num_samples

  \bug Objects which are only at the edge of the image can be missed
*/
void 
Shape3D::construct_volume(VoxelsOnCartesianGrid<float> &image, 
                          const CartesianCoordinate3D<int>& num_samples) const
{ 
  const CartesianCoordinate3D<float>& voxel_size= image.get_voxel_size();
  const CartesianCoordinate3D<float>& origin= image.get_origin();
  //if (norm(origin)>.00001)
  //    error("Shape3D::construct_volume currently ignores image origin (not shape origin)\n");
  const int min_z = image.get_min_z();
  const int min_y = image.get_min_y();
  const int min_x = image.get_min_x();
  const int max_z = image.get_max_z();
  const int max_y = image.get_max_y();
  const int max_x = image.get_max_x();

  CartesianCoordinate3D<int> crude_num_samples(1,1,1);

  for(int z = min_z;z<=max_z;z++)
  {
    for(int y =min_y;y<=max_y;y++)
      for(int x=min_x;x<=max_x;x++)
	
      {
        const CartesianCoordinate3D<float> 
	  current_index(static_cast<float>(z),
			static_cast<float>(y),
			static_cast<float>(x));
	
	//image[z][y][x] = get_voxel_weight(current_point,voxel_size,crude_num_samples);

	image[z][y][x] = 
	  (is_inside_shape(current_index*voxel_size+origin))
	  ? 1.F : 0.F;
      }
  }
      
   if (num_samples.x() == 1 && num_samples.y() == 1 && num_samples.z() == 1)
    return;

  int num_recomputed = 0;
  for(int z =min_z;z<=max_z;z++)  
    for(int y =min_y;y<=max_y;y++)
      for(int x=min_x;x<= max_x;x++)
      {
	const float current_value = image[z][y][x];

	// first check if we're already at an edge voxel
	// Note:  this allow fuzzy boundaries
	bool recompute = current_value<.999F && current_value>.00F;
	if (!recompute)
	  {
	    // check neighbour values. If they are all equal, we'll assume it's ok.
	    for(int i = z-1;!recompute && (i<=z+1);i++)
	      for(int j= y-1;!recompute && (j<=y+1);j++)
		for(int k=x-1;!recompute && (k<=x+1);k++)	      
		  {
		    const float value_of_neighbour =
		      ((i < min_z) || (i> max_z) ||
		       (j < min_y) || (j> max_y) ||
	               (k < min_x) || (k> max_x)
 	              ) ? 0 : image[i][j][k];
                      recompute =  (value_of_neighbour!=current_value);
                   } 
          }
        if (recompute)
	{
	  num_recomputed++;
	  const CartesianCoordinate3D<float> 
	    current_index(static_cast<float>(z),
			  static_cast<float>(y),
			  static_cast<float>(x));
	  image[z][y][x] = get_voxel_weight(current_index*voxel_size+origin,voxel_size,num_samples);
	}
      }
  cerr << "Number of voxels recomputed with finer sampling : " << num_recomputed << endl;
      
}

#if 0
void Shape3D::construct_slice(PixelsOnCartesianGrid<float> &plane, 
                              const CartesianCoordinate3D<int>& num_samples) const
 {

  // TODO
  // CartesianCoordinate3D<float> voxel_size = plane.get_voxel_size();
  CartesianCoordinate2D<float> pre_voxel_size= plane.get_pixel_size();
  //TODO
  //CartesianCoordinate3D<float> grid_spacing =plane.get_grid_spacing();
  int voxel_size_z=1;// grid_spacing.z();

  CartesianCoordinate3D<float> voxel_size(pre_voxel_size.x(), pre_voxel_size.y(),voxel_size_z);
  CartesianCoordinate3D<float> origin = plane.get_origin();
  int z = origin.z();

  //TODO
   for ( int y = plane.get_min_y(); y<=plane.get_max_y(); y++)
   for ( int x = plane.get_min_x();x<=plane.get_max_x();x++)
   
   { 
    CartesianCoordinate3D<float> current_point(x,y,z);
    plane[y][x]= get_voxel_weight(current_point,voxel_size, num_samples);
   }
 }
 
#endif

void 
Shape3D::
set_defaults()
{
  origin[3] = origin[2] = origin[1] =0;
}

void 
Shape3D::
initialise_keymap()
{
  this->parser.add_key("origin (in mm)", &origin);
}

std::string 
Shape3D::parameter_info()
{
  return ParsingObject::parameter_info();
}


END_NAMESPACE_STIR
