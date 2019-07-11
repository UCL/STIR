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
class zoom_imageAdjointTests : public RunTests
{
public:
  void run_tests();
  void fill_image_with_random(VoxelsOnCartesianGrid<float> & image);
  float dot_product(VoxelsOnCartesianGrid<float> & image1, VoxelsOnCartesianGrid<float> & image2);
};

void
zoom_imageAdjointTests::fill_image_with_random(VoxelsOnCartesianGrid<float> & image)
{
    for(int i=0 ; i<image.get_max_z() ; i++){
               for(int j=0 ; j<image.get_max_y(); j++){
                   for(int k=0 ; k<image.get_max_z() ; k++){

                       image[i][j][k] = rand()%10;

                   }
                 }
             }

}

float
zoom_imageAdjointTests::dot_product(VoxelsOnCartesianGrid<float> & image1,VoxelsOnCartesianGrid<float> & image2)
{
    float cdot  = 0;
    for(int i=0 ; i<image1.get_max_z() ; i++){
               for(int j=0 ; j<image1.get_max_y(); j++){
                   for(int k=0 ; k<image1.get_max_z() ; k++){

                       cdot += image1[i][j][k]*image2[i][j][k];

                   }
                 }
             }

        std::cout<< "Dot Product:" << cdot << '\n';
    return cdot;
}


void
zoom_imageAdjointTests::run_tests()

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

    VoxelsOnCartesianGrid<float>  A(new_range,new_origin, new_grid_spacing);
    VoxelsOnCartesianGrid<float>  At(range,origin, grid_spacing);

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


    fill_image_with_random(image);
    fill_image_with_random(new_image);

    //test preserve projections
    zoom_image(A, image,ZoomOptions::preserve_projections);
    transpose_zoom_image(At, new_image,ZoomOptions::preserve_projections);
    float cdot1 = dot_product(A,new_image);
    float cdot2 = dot_product(At,image);

    set_tolerance(0.004);
    check_if_equal(cdot1,cdot2,"test on zoom option : preserve_projections");

    //test preserve values
    zoom_image(A, image,ZoomOptions::preserve_values);
    transpose_zoom_image(At, new_image,ZoomOptions::preserve_values);
    cdot1 = dot_product(A,new_image);
    cdot2 = dot_product(At,image);

    set_tolerance(0.004);
    check_if_equal(cdot1,cdot2,"test on zoom option : preserve_values");

   //test preserve sum
    zoom_image(A, image,ZoomOptions::preserve_sum);
    transpose_zoom_image(At, new_image,ZoomOptions::preserve_sum);
    cdot1 = dot_product(A,new_image);
    cdot2 = dot_product(At,image);

    set_tolerance(0.004);
    check_if_equal(cdot1,cdot2,"test on zoom option : preserve_sum");




  }




}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main()
{
  zoom_imageAdjointTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
