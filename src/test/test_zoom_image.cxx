//
//
/*
    Copyright (C) 2006- 2007,  Hammersmith Imanet Ltd 
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
  \ingroup test
  
  \brief Test program for stir::zoom_image (and stir::centre_of_gravity)
    
  \author Kris Thielemans
  
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange.h"
#include "stir/zoom.h"
#include "stir/centre_of_gravity.h"

#include <iostream>
#include <math.h>
#include "stir/RunTests.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for zoom_image (and centre_of_gravity)

  The tests check if a point source remains in the same physical location
  after zooming. This is done by checking the centre of gravity of the
  zoomed image.
*/
class zoom_imageTests : public RunTests
{
public:
  void run_tests();
};


void
zoom_imageTests::run_tests()

{ 
  std::cerr << "Tests for zoom_image\n";
  
  CartesianCoordinate3D<float> origin (0,1,2);  
  CartesianCoordinate3D<float> grid_spacing (3,4,5); 
  
  IndexRange<3> 
    range(CartesianCoordinate3D<int>(0,-15,-14),
          CartesianCoordinate3D<int>(4,14,15));
  
  VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);
  image.fill(0.F);

  const BasicCoordinate<3,int> indices = make_coordinate(1,2,3);
  image[indices] = 1.F;
  const CartesianCoordinate3D<float> coord =
    image.get_physical_coordinates_for_indices(indices);

  {
    // check if centre_of_gravity_in_mm returns same point
    check_if_equal(coord, 
		   find_centre_of_gravity_in_mm(image),
		   "test on get_physical_coordinates_for_indices and find_centre_of_gravity_in_mm");
  }

  // we cannot have very good accuracy in the centre of gravity
  // calculation for zooming a single pixel
  // the following threshold seems very reasonable (.3mm distance) and works.
  const double tolerance_for_distance = .3;
  const double old_tolerance = this->get_tolerance();

  // test 2 arg zoom_image
  {
    CartesianCoordinate3D<float> new_origin (4.F,5.F,6.F);  
    CartesianCoordinate3D<float> new_grid_spacing (2.2F,3.1F,4.3F); 
  
    IndexRange<3> 
      new_range(CartesianCoordinate3D<int>(-1,-16,-17),
		CartesianCoordinate3D<int>(5,15,20));
    
    VoxelsOnCartesianGrid<float>  new_image(new_range,new_origin, new_grid_spacing);
    zoom_image(new_image, image);
    {
      // check if centre_of_gravity_in_mm returns same point
      this->set_tolerance(tolerance_for_distance);
      check_if_equal(coord, 
		     find_centre_of_gravity_in_mm(new_image),
		     "test on 2-argument zoom_image");
      this->set_tolerance(old_tolerance);
      check_if_equal(new_range, new_image.get_index_range(),
		     "test on 2-argument argument zoom_image: index range");
      check_if_equal(new_grid_spacing, new_image.get_voxel_size(), 
		     "test on 2-argument argument zoom_image: voxel size");
      check_if_equal(new_origin, new_image.get_origin(),
		     "test on 2-argument argument zoom_image: origin");

    }
  }


  // test multiple argument zoom_image
  {
    const CartesianCoordinate3D<float> zooms(1.3F,1.2F,1.5F);
    const CartesianCoordinate3D<float> offsets_in_mm(3.F,4.F,5.5F);
    const Coordinate3D<int> new_sizes(30,40,50);
    const VoxelsOnCartesianGrid<float>  new_image =
      zoom_image(image, zooms, offsets_in_mm, new_sizes);
    {
      // check if centre_of_gravity_in_mm returns same point
      this->set_tolerance(tolerance_for_distance);
      check_if_equal(coord, 
		     find_centre_of_gravity_in_mm(new_image),
		     "test on multiple argument zoom_image");
      this->set_tolerance(old_tolerance);
      check_if_equal(new_sizes, new_image.get_lengths(),
		     "test on multiple argument zoom_image: index range");
      check_if_equal(new_image.get_voxel_size(), image.get_voxel_size()/zooms,
		     "test on multiple argument zoom_image: voxel size");

    }
  }

  // test multiple argument zoom_image in 2D
  {
    const float zoom = 1.3F;
    const CartesianCoordinate3D<float> zooms(1.F,zoom,zoom);
    const CartesianCoordinate3D<float> offsets_in_mm(0.F,4.F,5.5F);
    const int new_size = 30;
    const VoxelsOnCartesianGrid<float>  new_image =
      zoom_image(image, zoom, offsets_in_mm.x(), offsets_in_mm.y(), new_size);
    {
      // check if centre_of_gravity_in_mm returns same point
      this->set_tolerance(tolerance_for_distance);
      check_if_equal(coord, 
		     find_centre_of_gravity_in_mm(new_image),
		     "test on multiple argument (2d) zoom_image");
      this->set_tolerance(old_tolerance);
      check_if_equal(image.get_min_z(), new_image.get_min_z(),
		     "test on multiple argument (2d) zoom_image: min_z");
      check_if_equal(image.get_max_z(), new_image.get_max_z(),
		     "test on multiple argument (2d) zoom_image: max_z");
      check_if_equal(new_size, new_image.get_x_size(), 
		     "test on multiple argument (2d) zoom_image: x_size");
      check_if_equal(new_size, new_image.get_y_size(), 
		     "test on multiple argument (2d) zoom_image: y_size");
      check_if_equal(new_image.get_voxel_size(), image.get_voxel_size()/zooms,
		     "test on multiple argument (2d) zoom_image: voxel size");

    }
  }

}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main()
{
  zoom_imageTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
