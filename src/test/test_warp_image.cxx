//
//
/*
 Copyright (C) 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
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
  \ingroup spatial_transformation
 
  \brief A simple program to test the warp image functions
  \author Charalampos Tsoumpas
*/

#include "stir/CartesianCoordinate3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/GatedDiscretisedDensity.h"
#include "stir/IndexRange.h"
#include "stir/spatial_transformation/warp_image.h"
#include "stir/RunTests.h"
#include "stir/spatial_transformation/GatedSpatialTransformation.h"
#include <iostream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

/*!
  \brief Class with tests for warp_image functions.
  \ingroup test
*/
class warp_imageTests : public RunTests
{
public:
  void run_tests();
};

void
warp_imageTests::run_tests()
{
  std::cerr << "Tests for warp_image" << std::endl;

  CartesianCoordinate3D<float> origin (0,1,2);  
  CartesianCoordinate3D<float> grid_spacing (3,4,5); 
  
  IndexRange<3> 
    range(CartesianCoordinate3D<int>(0,-15,-14),
          CartesianCoordinate3D<int>(40,44,45));
  
  VoxelsOnCartesianGrid<float>  image(range, origin, grid_spacing);
  image.fill(0.F);

  const BasicCoordinate<3,int> indices = make_coordinate(10,22,23);
  image[indices] = 1.F;

  VoxelsOnCartesianGrid<float>  motion_x(range, origin, grid_spacing);
  VoxelsOnCartesianGrid<float>  motion_y(range, origin, grid_spacing);
  VoxelsOnCartesianGrid<float>  motion_z(range, origin, grid_spacing);

  motion_x.fill(3*grid_spacing[3]);
  motion_y.fill(2*grid_spacing[2]);
  motion_z.fill(grid_spacing[1]);

  // horrible way - but it works. I need to make it simpler. 
  const shared_ptr<VoxelsOnCartesianGrid<float> > image_sptr(image.clone()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > motion_x_sptr(motion_x.clone()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > motion_y_sptr(motion_y.clone()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > motion_z_sptr(motion_z.clone()) ;
  std::vector<std::pair<unsigned int, double> > gate_sequence;
  gate_sequence.resize(2);
  for (unsigned int current_gate = 1; 
       current_gate <= 2; 
       ++current_gate)
    {
      gate_sequence[current_gate-1].first = current_gate;
      gate_sequence[current_gate-1].second = 1;
    }

  TimeGateDefinitions gate_defs(gate_sequence);

  const  VoxelsOnCartesianGrid<float> new_image=warp_image(image_sptr,motion_x_sptr,motion_y_sptr,motion_z_sptr,BSpline::BSplineType(1),0);
  const BasicCoordinate<3,int> new_indices = make_coordinate(indices[1]-1,indices[2]-2,indices[3]-3);

  {
    check_if_equal(image[indices],1.F, "testing original image at non-zero point");
    check_if_equal(image[new_indices],0.F, "testing original image at new location");
    check_if_equal(new_image[indices],0.F, "testing warped image at original location");
    check_if_equal(new_image[new_indices],1.F, "testing warped image at new location");
  }
  std::cerr << "Tests for class GatedSpatialTransformation::warp_image etc" << std::endl;
  const shared_ptr<VoxelsOnCartesianGrid<float> > new_image_sptr(new_image.clone()) ;
  GatedDiscretisedDensity gated_image(image_sptr,2);
  gated_image.set_density_sptr(image_sptr,1);
  gated_image.set_density_sptr(new_image_sptr,2);
  gated_image.set_time_gate_definitions(gate_defs);
  VoxelsOnCartesianGrid<float>  reverse_motion_x(range, origin, grid_spacing);
  VoxelsOnCartesianGrid<float>  reverse_motion_y(range, origin, grid_spacing);
  VoxelsOnCartesianGrid<float>  reverse_motion_z(range, origin, grid_spacing);
  reverse_motion_x.fill(-3*grid_spacing[3]);
  reverse_motion_y.fill(-2*grid_spacing[2]);
  reverse_motion_z.fill(-1*grid_spacing[1]);

  // horrible way - but it works. I need to make it simpler. 
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion1_x_sptr(motion_x.get_empty_copy()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion1_y_sptr(motion_y.get_empty_copy()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion1_z_sptr(motion_z.get_empty_copy()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion2_x_sptr(reverse_motion_x.clone()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion2_y_sptr(reverse_motion_y.clone()) ;
  const shared_ptr<VoxelsOnCartesianGrid<float> > reverse_motion2_z_sptr(reverse_motion_z.clone()) ;

  GatedDiscretisedDensity reverse_gated_motion_x(image_sptr,2);
  reverse_gated_motion_x.set_density_sptr(reverse_motion1_x_sptr,1);
  reverse_gated_motion_x.set_density_sptr(reverse_motion2_x_sptr,2);
  reverse_gated_motion_x.set_time_gate_definitions(gate_defs);

  GatedDiscretisedDensity reverse_gated_motion_y(image_sptr,2);
  reverse_gated_motion_y.set_density_sptr(reverse_motion1_y_sptr,1);
  reverse_gated_motion_y.set_density_sptr(reverse_motion2_y_sptr,2);
  reverse_gated_motion_y.set_time_gate_definitions(gate_defs);

  GatedDiscretisedDensity reverse_gated_motion_z(image_sptr,2);
  reverse_gated_motion_z.set_density_sptr(reverse_motion1_z_sptr,1);
  reverse_gated_motion_z.set_density_sptr(reverse_motion2_z_sptr,2);
  reverse_gated_motion_z.set_time_gate_definitions(gate_defs);
  
  GatedSpatialTransformation mvtest;
  mvtest.set_gate_defs(gate_defs);
  mvtest.set_spatial_transformations(reverse_gated_motion_z,reverse_gated_motion_y,reverse_gated_motion_x);
  VoxelsOnCartesianGrid<float> accumulated_image(range, origin, grid_spacing);
  mvtest.warp_image(accumulated_image,gated_image);
  {
    // simple test for gated_image values
    check_if_equal((gated_image.get_density(1))[indices],1.F, "testing 1st gate (i.e. original image) at non-zero point");
    check_if_equal((gated_image.get_density(1))[new_indices],0.F, "testing 1st gate at new location of the non-zero point");
    check_if_equal((gated_image.get_density(2))[indices],0.F, "testing 2nd gate at the original location of non-zero point");
    check_if_equal((gated_image.get_density(2))[new_indices],1.F, "testing 2nd gate at the new location of the non-zero point");
    check_if_equal((int)gated_image.get_time_gate_definitions().get_num_gates(),2, "testing gate_defs of gated_image are set correctly");

    // test if motion vectors have been set correctly
    check_if_equal((reverse_gated_motion_z.get_density(2))[indices],-1*grid_spacing[1], "testing the input to set the motion in z");
    check_if_equal((mvtest.get_spatial_transformation_z().get_density(2))[indices],-1*grid_spacing[1], "testing GatedSpatialTransformation class get the motion vector z correctly");
    check_if_equal((mvtest.get_spatial_transformation_z().get_density(2))[new_indices],-1*grid_spacing[1], "testing GatedSpatialTransformation class get the motion vector z correctly");
    check_if_equal((mvtest.get_spatial_transformation_y().get_density(2))[indices],-2*grid_spacing[2], "testing GatedSpatialTransformation class get the motion vector y correctly");
    check_if_equal((mvtest.get_spatial_transformation_y().get_density(2))[new_indices],-2*grid_spacing[2], "testing GatedSpatialTransformation class get the motion vector y correctly");
    check_if_equal((mvtest.get_spatial_transformation_x().get_density(2))[indices],-3*grid_spacing[3], "testing GatedSpatialTransformation class get the motion vector x correctly");
    check_if_equal((mvtest.get_spatial_transformation_x().get_density(2))[new_indices],-3*grid_spacing[3], "testing GatedSpatialTransformation class get the motion vector x correctly");
    check_if_equal((int)gate_defs.get_num_gates(),2, "testing gate_defs are set correctly");
    check_if_equal((int)reverse_gated_motion_z.get_time_gate_definitions().get_num_gates(),2, "testing GatedSpatialTransformation class get the motion vector x correctly");
    check_if_equal((int)(mvtest.get_time_gate_definitions()).get_num_gates(),2, "testing GatedSpatialTransformation class time_gate_difinitions");
    // actual test for accumulate_warp_image
    check_if_equal(accumulated_image[indices], 2.F, "testing the accumulated image at the original location of non-zero point");
    check_if_equal(accumulated_image[new_indices], 0.F, "testing the accumulated image at the location where the non-zero point had moved");
  }
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  warp_imageTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
