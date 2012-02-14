//
// $Id$
//
/*
    Copyright (C) 2002 - 2005-06-09, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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
  list_projdata_info [--all] projdata_filename
  </pre>
  Add the <tt>--all</tt> option to get min/max/sum information

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/SegmentByView.h"
#include "stir/is_null_ptr.h"
#include <iostream> 
#include <limits>
#include <string>

USING_NAMESPACE_STIR

void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<<"Usage: " << program_name << " [--all] projdata_file\n"
	   <<"\nAdd the --all option to get min/max/sum information\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{ 
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  // default value
  bool print_all = false;

  // first process command line options
  while (argc>0 && argv[0][0]=='-' && argc>=2)
    {
      if (strcmp(argv[0], "--all")==0)
	{
	  print_all = true; 
	  --argc; ++argv;
	}
      else
	print_usage_and_exit(program_name);
    }

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

  std::cout << "Info for file " << filename << '\n';
  std::cout << proj_data_sptr->get_proj_data_info_ptr()->parameter_info() << std::endl;

  if (print_all)
    {
      const int min_segment_num = proj_data_sptr->get_min_segment_num();
      const int max_segment_num = proj_data_sptr->get_max_segment_num();     
      bool accumulators_initialized = false;
      float accum_min=std::numeric_limits<float>::max(); // initialize to very large in case projdata is empty (although that's unlikely)
      float accum_max=std::numeric_limits<float>::min();
      double sum=0.;
      for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num) 
	{
	    const SegmentByView<float> seg(proj_data_sptr->get_segment_by_view(segment_num));
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
      std::cout << "\nData min: " << accum_min
		<< "\nData max: " << accum_max
		<< "\nData sum: " << sum
		<< "\n";
    }
  return EXIT_SUCCESS;
}
