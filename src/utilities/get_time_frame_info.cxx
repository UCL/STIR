//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief Prints start time and duration of a frame to stdout
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  cerr << "Usage:\n" << prog_name << "\\\n"
       << "\t\t[--msecs] [--duration | --start-time | --end-time] \\\n"
       << "\t\tFrame_def_filename Frame_number\n"
       << "Use the --duration option to get only the frame_duration on stdout,\n"
       << "and similarly for --start-time and --end-time.\n"
       << "Without these options, both will be printed to stdout, together with some text.\n\n"
       << "Times are reported in seconds unless the --msecs option is used."<< endl;
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

  while (argc>3)
    {
      if (strcmp(argv[1], "--msecs")==0)
	{
	  units_secs=false;
	  --argc; ++argv;
	}
      else 
	{
	  if (strcmp(argv[1], "--duration")==0 && !only_start_time && !only_end_time)
	    {
	      only_duration = true;
	      --argc; ++argv;
	    } 
	  else if (strcmp(argv[1], "--start-time")==0 && !only_duration && !only_end_time)
	    {
	      only_start_time = true;
	      --argc; ++argv;
	    }
	  else if (strcmp(argv[1], "--end-time")==0 && !only_duration && !only_start_time)
	    {
	      only_end_time = true;
	      --argc; ++argv;
	    }
	  else
	    print_usage_and_exit(prog_name);
	}
    }
  if(argc !=3)
    print_usage_and_exit(prog_name);

  
  const TimeFrameDefinitions time_def(argv[1]);
  const unsigned int frame_num = atoi(argv[2]);

  if (frame_num > time_def.get_num_frames() || frame_num<1)
    {
      warning("frame_num should be between 1 and %d\n", 
	      time_def.get_num_frames());
      exit(EXIT_FAILURE);
    }
  const double start_frame = time_def.get_start_time(frame_num);
  const double end_frame = time_def.get_end_time(frame_num);
  const double frame_duration = end_frame-start_frame;

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
  else
    cout << "Start of frame : " << start_frame*units << units_string 
	 << "\nEnd of frame   : " << end_frame*units << units_string 
	 << "\nFrame duration : " << frame_duration*units << units_string << endl;



  return EXIT_SUCCESS;
}

