//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities

  \brief Takes the logarithm of attenuation coefficients to 'convert' them to line integrals.

  \par Usage
  \verbatim
  attenuation_coefficients_to_projections \
       --AF|--ACF <output filename > <input proj_data file name>
  \endverbatim
  Use <tt>--AF</tt> if input are attenuation factors, <tt>--ACF</tt> for 
  attenuation correction factors (i.e. the inverse of the former).

  \warning Currently thresholds ACF values to maximum 150 (and AF to minimum 1/150)
   to prevent log() to grow too large.
  \todo get threshold from command line

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/


#include "stir/ProjData.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"
#include "stir/thresholding.h"
#include <iostream>

USING_NAMESPACE_STIR

static void print_usage_and_exit()
{
  std::cerr<<"\nUsage:\nattenuation_coefficients_to_projections\n\t"
	   << " --AF|--ACF <output filename > <input proj_data file name>  \n";
  exit(EXIT_FAILURE);
}

int 
main (int argc, char * argv[])
{

  // TODO get this from cmdline
  const float acf_threshold=150.F;
  
  if (argc!=4)
    print_usage_and_exit();

  bool doACF;
  if (strcmp(argv[1],"--ACF")==0)
    doACF=true;
  else if (strcmp(argv[1],"--AF")==0)
    doACF=false;
  else
    print_usage_and_exit();

  ++argv; --argc;
  
  shared_ptr <ProjData> attenuation_proj_data_ptr =
    ProjData::read_from_file(argv[2]);

  const string output_file_name = argv[1];

  shared_ptr<ProjData> out_proj_data_ptr =
    new ProjDataInterfile(attenuation_proj_data_ptr->get_proj_data_info_ptr()->clone(),
			  output_file_name);

  for (int segment_num = attenuation_proj_data_ptr->get_min_segment_num(); 
       segment_num<= attenuation_proj_data_ptr->get_max_segment_num();
       ++segment_num)
    for ( int view_num = attenuation_proj_data_ptr->get_min_view_num();
	  view_num<=attenuation_proj_data_ptr->get_max_view_num(); 
	  ++view_num)
    {
      Viewgram<float> viewgram = attenuation_proj_data_ptr->get_viewgram(view_num,segment_num);
      
      if (doACF)
	{
	  // threshold minimum to arbitrary value as log will otherwise explode)
	  threshold_lower(viewgram.begin_all(), viewgram.end_all(), 1/acf_threshold);
	  in_place_log(viewgram);
	}
      else
	{
	  // threshold maximum to arbitrary value as log will otherwise explode)
	  threshold_upper(viewgram.begin_all(), viewgram.end_all(), acf_threshold);
	  in_place_log(viewgram);
	  viewgram *= -1.F;
	}

      if (out_proj_data_ptr->set_viewgram(viewgram) != Succeeded::yes)
	{
	  warning("Error setting output viewgram at segment %d view %d. Exiting",
		  segment_num, view_num);
	  return EXIT_FAILURE;
	}
    }
    
  
  return EXIT_SUCCESS;
}

