/*!
  \file
  \ingroup examples
  \brief A simple program that creates an image and fills it with a shape

  Code is loosely based on GenerateImage.cxx
  \author Kris Thielemans
*/
/*
    Copyright (C) 2018-2022, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/PatientPosition.h"
#include "stir/ImagingModality.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_to_file.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <iostream>

int main()
{

  std::string output_filename("test.hv");
  int output_image_size_x = 100;
  int output_image_size_y = 104;
  int output_image_size_z = 20;
  float output_voxel_size_x = 5.F;
  float output_voxel_size_y = 4.F;
  float output_voxel_size_z = 3.F;
  double image_duration = 100.; // secs
  double rel_start_time = 1; //secs

  auto exam_info_sptr = std::make_shared<stir::ExamInfo>(stir::ImagingModality::PT);
  {
    stir::TimeFrameDefinitions frame_defs;
    frame_defs.set_num_time_frames(1);
    frame_defs.set_time_frame(1, rel_start_time, image_duration);
    exam_info_sptr->set_time_frame_definitions(frame_defs);
  }
  stir::VoxelsOnCartesianGrid<float>
          image(exam_info_sptr,
                        stir::IndexRange3D(0,output_image_size_z-1,
                                     -(output_image_size_y/2),
                                     -(output_image_size_y/2)+output_image_size_y-1,
                                     -(output_image_size_x/2),
                                     -(output_image_size_x/2)+output_image_size_x-1),
                        stir::CartesianCoordinate3D<float>(0,0,0),
                        stir::CartesianCoordinate3D<float>(output_voxel_size_z,
                                                     output_voxel_size_y,
                                                     output_voxel_size_x));

  // add shape to image
  {
    const auto centre =
      image.get_physical_coordinates_for_indices(stir::make_coordinate(output_image_size_z/2,0,0));
    stir::EllipsoidalCylinder shape(40.F, 30.F, 20.F, centre);
    shape.construct_volume(image, stir::make_coordinate(2,2,2));
  }

  // write output for checking
  try
    {
      stir::write_to_file(output_filename, image);
      stir::info("Image written as " + output_filename);
    }
  catch (...)
    {
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;  
}
