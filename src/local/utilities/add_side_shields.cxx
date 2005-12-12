//
// $Id$
//
/*
  Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \ingroup utilities

  \brief A preliminary utility to add side shields to an attenuation image.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/Shape/EllipsoidalCylinder.h"
//#include "stir/Shape/CombinedShape3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <iostream>




int main(int argc, char * argv[])
{

  USING_NAMESPACE_STIR;
  using namespace std;

  if ( argc!=3) {
    cerr << "Usage: " << argv[0] << " output_filename input_filename\n";
    exit(EXIT_FAILURE);
  }

  const char * const output_filename = argv[1];
  const char * const input_filename = argv[2];
  shared_ptr<DiscretisedDensity<3,float> > density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);
  VoxelsOnCartesianGrid<float> current_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>& >(*density_ptr);

  const float distance_of_shield_inner_to_centre = 78.35F;
  const float distance_of_shield_outer_to_centre = 150.F;
  const float shield_thickness = 
    distance_of_shield_outer_to_centre - distance_of_shield_inner_to_centre;
  const float shield_outer_radius = 433.F;
  const float shield_inner_radius = 379.F;
  const float mu_value_for_shield = 1.76568F; // in cm^-1
  const CartesianCoordinate3D<float> 
    front_shield_centre(-(distance_of_shield_inner_to_centre+shield_thickness/2),
			0,
			0);
  const CartesianCoordinate3D<float> 
    back_shield_centre(+(distance_of_shield_inner_to_centre+shield_thickness/2),
			0,
			0);

  shared_ptr<Shape3D>
    front_shield_outer_sptr =
    new EllipsoidalCylinder(shield_thickness, shield_outer_radius, shield_outer_radius,
			    front_shield_centre,
			    0,0,0);
  shared_ptr<Shape3D>
    front_shield_inner_sptr =
    new EllipsoidalCylinder(shield_thickness, shield_inner_radius, shield_inner_radius,
			    front_shield_centre,
			    0,0,0);
  //  CombinedShape3D<logical_and_not<bool> > front_shield(front_shield_outer_sptr,
  //						       front_shield_inner_sptr);
  shared_ptr<Shape3D>
    back_shield_outer_sptr =
    new EllipsoidalCylinder(shield_thickness, 433, 433,
			    back_shield_centre,
			    0,0,0);
  shared_ptr<Shape3D>
    back_shield_inner_sptr = 
    new EllipsoidalCylinder(shield_thickness, 379, 379,
			    back_shield_centre,
			    0,0,0);
  //CombinedShape3D<logical_and_not<bool> > back_shield(back_shield_outer_sptr,
  //						      back_shield_inner_sptr);


  
  BasicCoordinate<3,int> min_indices, max_indices;
  current_image.get_regular_range(min_indices, max_indices);
  const CartesianCoordinate3D<float> voxel_size = current_image.get_voxel_size();
  const CartesianCoordinate3D<float> origin = current_image.get_origin();
  const float shift_z_to_centre = 
    (min_indices[1]+max_indices[1])/2.F*voxel_size[1];

  const int old_num_planes = current_image.size();
  if (old_num_planes%2==0)
    error("Cannot handle odd number of planes in input image yet");

  int new_num_planes = 
    std::max(current_image.get_z_size(),
	     static_cast<int>(ceil(2* distance_of_shield_outer_to_centre / voxel_size.z())));
  if (new_num_planes%2==0)
    ++new_num_planes;

  const int old_index_to_new = 
    old_num_planes/2 - new_num_planes/2;

  int new_max_xy = 
    std::max(current_image[0].size()/2+1,
	     std::max(current_image[0][0].size()/2+1,
		      std::max(static_cast<size_t>(ceil(shield_outer_radius / voxel_size.x())),
			       static_cast<size_t>(ceil(shield_outer_radius / voxel_size.y())))));
  //const BasicCoordinate<3,int> new_min_indices = 
  //  make_coord(old_index_to_new, -new_max_xy, -new_max_xy);
  //const BasicCoordinate<3,int> new_max_indices = 
  //  make_coord(new_num_planes + old_index_to_new - 1, new_max_xy, new_max_xy);
  const Coordinate3D<int> new_min_indices (old_index_to_new, -new_max_xy, -new_max_xy);
  const Coordinate3D<int> new_max_indices (new_num_planes + old_index_to_new - 1, new_max_xy, new_max_xy);

  CartesianCoordinate3D<float> new_origin = origin;
  new_origin.z() -= shift_z_to_centre;

  VoxelsOnCartesianGrid<float> output_image(IndexRange<3>(new_min_indices, new_max_indices),
					    new_origin,
					    voxel_size);

  VoxelsOnCartesianGrid<float> temp_image(IndexRange<3>(new_min_indices, new_max_indices),
					  new_origin,
					  voxel_size);
  
  output_image.fill(0);
  //const BasicCoordinate<3,int> num_samples = make_coord(5,5,5);
  const Coordinate3D<int> num_samples(5,5,5);
  //const Coordinate3D<int> num_samples(1,1,1);
  
  //front_shield.construct_volume(output_image, num_samples);
  //back_shield.construct_volume(temp_image, num_samples);
  //output_image += temp_image;

  front_shield_outer_sptr->construct_volume(output_image, num_samples);
  temp_image.fill(0);
  back_shield_outer_sptr->construct_volume(temp_image, num_samples);
  output_image += temp_image;
  temp_image.fill(0);
  front_shield_inner_sptr->construct_volume(temp_image, num_samples);
  output_image -= temp_image;
  temp_image.fill(0);
  back_shield_inner_sptr->construct_volume(temp_image, num_samples);
  output_image -= temp_image;

  output_image *= mu_value_for_shield;

  // now fill in old data
  for (int z=current_image.get_min_index(); z<= current_image.get_max_index(); ++z)
    for (int y=current_image[z].get_min_index(); y<= current_image[z].get_max_index(); ++y)
      for (int x=current_image[z][y].get_min_index(); x<= current_image[z][y].get_max_index(); ++x)
	output_image[z][y][x] += current_image[z][y][x];


  DefaultOutputFileFormat output_file_format;
  Succeeded success =
    output_file_format.write_to_file(output_filename, output_image);  
  
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
