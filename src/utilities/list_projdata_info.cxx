/*
    Copyright (C) 2002 - 2005-06-09, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, University College London
    Copyright (C) 2016, University of Hull
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

  \brief A utility that lists size info on the projection data on stdout.

  \par Usage

  <pre>
  list_projdata_info [--all  | --min | --max | --sum | --geom | --exam] projdata_filename
  </pre>
  Add one or more options to print the exam/geometric/min/max/sum information.
  If no option is specified, geometric info is printed.

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/ProjData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/SegmentByView.h"
#include "stir/is_null_ptr.h"
#include <iostream> 
#include <limits>
#include <string>

USING_NAMESPACE_STIR

void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<<"Usage: " << program_name << " [--all | --min | --max | --sum | --geom | --exam] projdata_file\n"
	   <<"\nAdd one or more options to print the exam/geometric/min/max/sum information.\n"
	   <<"\nIf no option is specified, geometric info is printed.\n";
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
    print_geom = true;

  if(argc!=1) 
  {
    print_usage_and_exit(program_name);
  }

  // set filename to last remaining argument
  const std::string filename(argv[0]);

  shared_ptr<ProjData> proj_data_sptr(ProjData::read_from_file(filename));

  if (is_null_ptr(proj_data_sptr))
    {
      warning("Could not read %s", filename.c_str());
      return EXIT_FAILURE;
    }

  if (print_exam)
    {
      const ExamInfo& exam_info = *proj_data_sptr->get_exam_info_ptr();
      std::cout << "Modality: " << proj_data_sptr->get_exam_info_ptr()->imaging_modality.get_name() << '\n';
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
    std::cout << proj_data_sptr->get_proj_data_info_ptr()->parameter_info() << std::endl;

  //if (print_min || print_max || print_sum)
  {
      const int min_segment_num = proj_data_sptr->get_min_segment_num();
      const int max_segment_num = proj_data_sptr->get_max_segment_num();
      const int min_timing_num = proj_data_sptr->get_min_timing_pos_num();
      const int max_timing_num = proj_data_sptr->get_max_timing_pos_num();
      std::cout << "\nTotal number of timing positions: " << proj_data_sptr->get_num_timing_poss();

      for (int timing_num = min_timing_num; timing_num <= max_timing_num; ++timing_num)
      {
          std::cout << "\nTiming location: " << timing_num;
          bool accumulators_initialized = false;
          float accum_min=std::numeric_limits<float>::max(); // initialize to very large in case projdata is empty (although that's unlikely)
          float accum_max=std::numeric_limits<float>::min();
          double sum=0.;
          for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
          {
              const SegmentByView<float> seg(proj_data_sptr->get_segment_by_view(segment_num, timing_num));
              const float this_max=seg.find_max();
              const float this_min=seg.find_min();
              sum+=static_cast<double>(seg.sum());
              if(!accumulators_initialized)
              {
                  accum_max=this_max;
                  accum_min=this_min;
                  accumulators_initialized=true;
              }
              else
              {
                  if (accum_max<this_max) accum_max=this_max;
                  if (accum_min>this_min) accum_min=this_min;
              }
          }
          if (print_min)
              std::cout << "\nData min: " << accum_min;
          if (print_max)
              std::cout << "\nData max: " << accum_max;
          if (print_sum)
              std::cout << "\nData sum: " << sum;
          std::cout << "\n";
      }
  }
  return EXIT_SUCCESS;
}
