//
//
/*
    Copyright (C) 2006- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2018, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief  This program lists basic image info

  \author Thielemans


  \warning It only supports stir::VoxelsOnCartesianGrid type of images.
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "stir/unique_ptr.h"
#include <memory>
#include <iostream>
USING_NAMESPACE_STIR

static void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<<"Usage: " << program_name << " [--all | --min | --max | --sum | --exam] image_file\n"
	   <<"\nAdd one or more options to print the exam/geometric/min/max/sum information.\n"
	   <<"\nIf no option is specified, geometric/min/max/sum info is printed.\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  // default values
  bool print_exam = false;
  bool print_geom = false;
  bool print_min = false;
  bool print_max = false;
  bool print_sum = false;
  bool no_options = true; // need this for default behaviour

  // first process command line options
  while (argc>0 && argv[0][0]=='-' && argc>=2)
    {
      no_options=false;
      if (strcmp(argv[0], "--all")==0)
	{
	  print_min = print_max = print_sum = print_geom = print_exam = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--max")==0)
	{
	  print_max = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--min")==0)
	{
	  print_min = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--sum")==0)
	{
	  print_sum = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--geom")==0)
	{
	  print_geom = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[0], "--exam")==0)
	{
	  print_exam = true;
	  --argc; ++argv;
	}
      else
	print_usage_and_exit(program_name);
    }
  if (no_options)
    {
      print_geom = true;
      print_min = true;
      print_max = true;
      print_sum = true;
    }

  if(argc!=1)
  {
    print_usage_and_exit(program_name);
  }

  // set filename to last remaining argument
  const std::string filename(argv[0]);
  unique_ptr<VoxelsOnCartesianGrid<float> >image_aptr
    (dynamic_cast<VoxelsOnCartesianGrid<float> *>(
	DiscretisedDensity<3,float>::read_from_file(filename))
    );

  BasicCoordinate<3,int> min_indices, max_indices;
  if (!image_aptr->get_regular_range(min_indices, max_indices))
    error("Non-regular range of coordinates. That's strange.\n");

  BasicCoordinate<3,float> edge_min_indices(min_indices), edge_max_indices(max_indices);
  edge_min_indices-= 0.5F;
  edge_max_indices+= 0.5F;

  if (print_exam)
    {
      const ExamInfo& exam_info = *image_aptr->get_exam_info_ptr();
      std::cout << "Modality: " << exam_info.imaging_modality.get_name() << '\n';
      std::cout << "Patient position: " << exam_info.patient_position.get_position_as_string() << '\n';
      std::cout << "Scan start time in secs since 1970 UTC: " << exam_info.start_time_in_secs_since_1970 << '\n';
      if (exam_info.time_frame_definitions.get_num_time_frames() == 1)
	{
	  std::cout << "Time frame start - end (duration), all in secs: "
		    << exam_info.time_frame_definitions.get_start_time(1)
		    << " - "
		    << exam_info.time_frame_definitions.get_end_time(1)
		    << " ("
		    << exam_info.time_frame_definitions.get_duration(1)
		    << ")\n";
	}
    }

  if (print_geom)
    std::cout << "\nOrigin in mm {z,y,x}    :" << image_aptr->get_origin()
              << "\nVoxel-size in mm {z,y,x}:" << image_aptr->get_voxel_size()
              << "\nMin_indices {z,y,x}     :" << min_indices
              << "\nMax_indices {z,y,x}     :" << max_indices
              << "\nNumber of voxels {z,y,x}:" << max_indices - min_indices + 1
              << "\nPhysical coordinate of first index in mm {z,y,x} :"
              << image_aptr->get_physical_coordinates_for_indices(min_indices)
              << "\nPhysical coordinate of last index in mm {z,y,x}  :"
              << image_aptr->get_physical_coordinates_for_indices(max_indices)
              << "\nPhysical coordinate of first edge in mm {z,y,x} :"
              << image_aptr->get_physical_coordinates_for_indices(edge_min_indices)
              << "\nPhysical coordinate of last edge in mm {z,y,x}  :"
              << image_aptr->get_physical_coordinates_for_indices(edge_max_indices);
  if (print_min)
    std::cout << "\nImage min: " << image_aptr->find_min();
  if (print_max)
    std::cout << "\nImage max: " << image_aptr->find_max();
  if (print_sum)
    std::cout<< "\nImage sum: " << image_aptr->sum();
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
