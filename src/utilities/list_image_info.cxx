//
//
/*
    Copyright (C) 2006- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2018, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief  This program lists basic image info. It works for dynamic images.

  Exam info and numerical info can be listed, depending on command line options. Run the utility to get a usage message.

  \author Kris Thielemans

  \warning It only supports stir::VoxelsOnCartesianGrid type of images.
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_from_file.h"
#include "stir/date_time_functions.h"
#include <memory>
#include <iostream>

USING_NAMESPACE_STIR

static void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<<"Usage: " << program_name << " [--all | --min | --max | --sum | --exam | --per-volume] image_file\n"
	   <<"\nAdd one or more options to print the exam/geometric/min/max/sum information.\n"
	   <<"\nIf no option is specified, geometric/min/max/sum info is printed."
           <<"For dynamic images, overall min/max/sum are printed, unless the --per-volume option is specified.\n";
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
  bool print_per_volume = false;
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
      else if (strcmp(argv[0], "--per-volume")==0)
	{
	  print_per_volume = true;
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
  auto image_aptr = read_from_file<DynamicDiscretisedDensity>(filename);

  if (print_exam)
    std::cout << image_aptr->get_exam_info_sptr()->parameter_info();

  if (print_geom)
    {
      BasicCoordinate<3,int> min_indices, max_indices;
      auto vox = dynamic_cast<VoxelsOnCartesianGrid<float> &>( image_aptr->get_density(1));
      if (!vox.get_regular_range(min_indices, max_indices))
        error("Non-regular range of coordinates. That's strange.");

      BasicCoordinate<3,float> edge_min_indices(min_indices), edge_max_indices(max_indices);
      edge_min_indices-= 0.5F;
      edge_max_indices+= 0.5F;

      std::cout << "\nOrigin in mm {z,y,x}    :" << vox.get_origin()
                << "\nVoxel-size in mm {z,y,x}:" << vox.get_voxel_size()
                << "\nMin_indices {z,y,x}     :" << min_indices
                << "\nMax_indices {z,y,x}     :" << max_indices
                << "\nNumber of voxels {z,y,x}:" << max_indices - min_indices + 1
                << "\nPhysical coordinate of first index in mm {z,y,x} :"
                << vox.get_physical_coordinates_for_indices(min_indices)
                << "\nPhysical coordinate of last index in mm {z,y,x}  :"
                << vox.get_physical_coordinates_for_indices(max_indices)
                << "\nPhysical coordinate of first edge in mm {z,y,x} :"
                << vox.get_physical_coordinates_for_indices(edge_min_indices)
                << "\nPhysical coordinate of last edge in mm {z,y,x}  :"
                << vox.get_physical_coordinates_for_indices(edge_max_indices);
    }

  if (print_per_volume)
    {
      for (unsigned vol = 1U; vol <= image_aptr->get_num_time_frames(); ++vol)
        {
          auto& volume = image_aptr->get_density(vol);
          if (print_min)
            std::cout << "\nImage " << vol << " min: " << *std::min_element(volume.begin_all_const(), volume.end_all_const());
          if (print_max)
            std::cout << "\nImage " << vol << " max: " << *std::max_element(volume.begin_all_const(), volume.end_all_const());
          if (print_sum)
            std::cout<< "\nImage " << vol << " sum: " << std::accumulate(volume.begin_all_const(), volume.end_all_const(), 0.F);
          std::cout << std::endl;
        }
    }
  else
    {    
      if (print_min)
        std::cout << "\nImage min: " << *std::min_element(image_aptr->begin_all_const(), image_aptr->end_all_const());
      if (print_max)
        std::cout << "\nImage max: " << *std::max_element(image_aptr->begin_all_const(), image_aptr->end_all_const());
      if (print_sum)
        std::cout<< "\nImage sum: " << std::accumulate(image_aptr->begin_all_const(), image_aptr->end_all_const(), 0.F);
      std::cout << std::endl;
    }
  return EXIT_SUCCESS;
}
