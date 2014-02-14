//
//

/*!
  \file
  \ingroup utilities

  \brief Prints start time and duration of a frame to stdout
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
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
#include "stir/TimeFrameDefinitions.h"
#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
#endif

USING_NAMESPACE_STIR

void print_usage_and_exit(char const * const prog_name)
{
  cerr << "Usage:\n" << prog_name << " PARAMETERS\n"
       << "where PARAMETERS has two possibilities:\n\n"
       << "1) print number of time frames\n"
       << "\t--num-time-frames Frame_def_filename\n\n"
       << "2) print info for one or more time frames\n"
       << "\t[--msecs] \\\n"
       << "\t[--duration | --start-time | --end-time | --mid-time] \\\n"
       << "\tFrame_def_filename start_frame_number [end_frame_number]\n\n"
       << "Use the --duration option to get only the frame_duration on stdout,\n"
       << "and similarly for --start-time, --mid-time and --end-time.\n"
       << "Without these options, both will be printed to stdout, together with some text.\n\n"
       << "Times are reported in seconds unless the --msecs option is used.\n\n"
       << "end_frame_number defaults to start_frame_number to print just a single frame."<< endl;
    exit(EXIT_FAILURE);
}

int
main(int argc, char* argv[])
{
  const char * const prog_name = argv[0];

  bool units_secs = true;
  bool only_duration = false;
  bool only_start_time = false;
  bool only_end_time = false;
  bool only_mid_time = false;
  bool only_num_time_frames = false;

  while (argc>=2 && argv[1][1]=='-')
    {
      if (strcmp(argv[1], "--num-time-frames")==0)
	{
	  only_num_time_frames = true;
	  --argc; ++argv;
	}
      else
	{
	  if (strcmp(argv[1], "--msecs")==0)
	    {
	      units_secs=false;
	      --argc; ++argv;
	    }
	  else if (strcmp(argv[1], "--duration")==0 && !only_start_time && !only_end_time && !only_mid_time)
	    {
	      only_duration = true;
	      --argc; ++argv;
	    } 
	  else if (strcmp(argv[1], "--start-time")==0 && !only_duration && !only_end_time && !only_mid_time)
	    {
	      only_start_time = true;
	      --argc; ++argv;
	    }
	  else if (strcmp(argv[1], "--end-time")==0 && !only_duration && !only_start_time && !only_mid_time)
	    {
	      only_end_time = true;
	      --argc; ++argv;
	    }
	  else if (strcmp(argv[1], "--mid-time")==0 && !only_duration && !only_start_time && !only_end_time)
	    {
	      only_mid_time = true;
	      --argc; ++argv;
	    }
	  else
	    print_usage_and_exit(prog_name);
	}
    }

  // we need at least one argument: the filename
  if(argc <2)
    print_usage_and_exit(prog_name);

  const TimeFrameDefinitions time_def(argv[1]);

  if (only_num_time_frames)
    {
      if(argc !=2)
	  print_usage_and_exit(prog_name);
      cout << time_def.get_num_frames() << std::endl;
      exit(EXIT_SUCCESS);
    }

  // normal case of info for one or more frames
  if(argc !=3 && argc!=4)
    print_usage_and_exit(prog_name);
  const unsigned int start_frame_num = atoi(argv[2]);
  const unsigned int end_frame_num = 
    argc>3 ? atoi(argv[3]) : start_frame_num;


  for (unsigned frame_num = start_frame_num; frame_num<=end_frame_num; ++frame_num)
    {
      if (frame_num > time_def.get_num_frames() || frame_num<1)
	{
	  /* Note: we intentionally check this in the loop.
	     This way, we do get output for valid frames.
	  */
	  warning("frame_num should be between 1 and %d\n", 
		  time_def.get_num_frames());
	  exit(EXIT_FAILURE);
	}
      const double start_frame = time_def.get_start_time(frame_num);
      const double end_frame = time_def.get_end_time(frame_num);
      const double frame_duration = end_frame-start_frame;
      const double mid_frame = (start_frame+end_frame)/2;

      // make sure results never get printed as scientific
      // because we pass it on to header_doc/edit_ecat_header
      // which expects an int
      cout.setf(std::ios::fixed, std::ios::floatfield);

      const int units = units_secs ? 1 : 1000;
      const std::string units_string = units_secs ? " secs" : " millisecs";
      if (only_duration)
	cout << frame_duration*units << endl;
      else if (only_start_time)
	cout << start_frame*units << endl;
      else if (only_end_time)
	cout << end_frame*units << endl;
      else if (only_end_time)
	cout << end_frame*units << endl;
      else if (only_mid_time)
	cout << mid_frame*units << endl;
      else
	cout << "Start of frame " << frame_num << "   : " << start_frame*units << units_string 
	     << "\nMiddle of frame " << frame_num << "  : " << mid_frame*units << units_string 
	     << "\nEnd of frame " << frame_num << "     : " << end_frame*units << units_string 
	     << "\nFrame duration " << frame_num << "   : " << frame_duration*units << units_string << endl;


    }
  return EXIT_SUCCESS;
}

