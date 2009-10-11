//
// $Id$
//
/*
  Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \brief It produces the absolute value image of an image. 
  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \todo Potentially, in the future it should be included in stir_math as an option.

  \par Usage:
  \code
  abs_image [-p || -d] -o output_filename -i input_filename 
  \endcode
  Use <tt>-p</tt> switch for parametric images, or the <tt>-d</tt> switch for dynamic images.
*/
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include <algorithm>
#include <getopt.h>

int main(int argc, char * argv[])
{
  USING_NAMESPACE_STIR;
  const char * output_filename = 0;
  const char * input_filename = 0;
  bool do_parametric=0;
  bool do_dynamic=0;

  const char * const usage = "abs_image [ -p | -d ] -o output_filename -i input_filename\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt (argc, argv, "i:o:(p||d)")) != -1)
      switch (c)
	{
	case 'i':
	  input_filename = optarg;
	  break;
	case 'o':
	  output_filename = optarg;
	  break;
	case 'p':
	  do_parametric=true;
	  break;
	case 'd':
	  do_dynamic=true;
	  break;
	case '?':
	  std::cerr << usage;
	  return EXIT_FAILURE;
	default:
	  if (isprint (optopt))
	    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	  else
	    fprintf (stderr,
		     "Unknown option character `\\x%x'.\n",
		     optopt);
	  std::cerr << usage;
	  return EXIT_FAILURE;
	}
  }

  if (output_filename==0 || input_filename==0)
    {
      std::cerr << usage;
      return EXIT_FAILURE;
    }

  if(do_parametric)// A better way will be to template it...
    {
      shared_ptr<ParametricVoxelsOnCartesianGrid> input_image_sptr = 
	ParametricVoxelsOnCartesianGrid::read_from_file(input_filename);  
      shared_ptr<ParametricVoxelsOnCartesianGrid> output_image_sptr = input_image_sptr->clone();		
      ParametricVoxelsOnCartesianGrid::full_iterator out_iter = output_image_sptr->begin_all();
      ParametricVoxelsOnCartesianGrid::const_full_iterator in_iter = input_image_sptr->begin_all_const();
      while( in_iter != input_image_sptr->end_all_const())
	{
	  if (*in_iter<0.F)
	    *out_iter = -(*in_iter);
	  ++in_iter; ++out_iter;
	}  
      Succeeded success =
	OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->
	write_to_file(output_filename, *output_image_sptr);
      return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
    }  
  else if(do_dynamic)// A better way will be to template it...
    {
      const shared_ptr<DynamicDiscretisedDensity> input_image_sptr =
	DynamicDiscretisedDensity::read_from_file(input_filename);  
      const DynamicDiscretisedDensity input_image = *input_image_sptr;
      DynamicDiscretisedDensity output_image = input_image;
      for(unsigned int frame_num=1;frame_num<=(input_image.get_time_frame_definitions()).get_num_frames();++frame_num)
	{
	  DiscretisedDensity<3,float>::full_iterator out_iter = output_image[frame_num].begin_all();
	  DiscretisedDensity<3,float>::const_full_iterator in_iter = input_image[frame_num].begin_all_const();
	  while( in_iter != input_image[frame_num].end_all_const())
	    {
	      if (*in_iter<0.F)
		*out_iter = -(*in_iter);
	      ++in_iter; ++out_iter;
	    }  
	}  
      Succeeded success =
	output_image.write_to_ecat7(output_filename);
      return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
    }
  else
    {
      shared_ptr<DiscretisedDensity<3,float> > input_image_sptr = 
	DiscretisedDensity<3,float>::read_from_file(input_filename);

      shared_ptr<DiscretisedDensity<3,float> > output_image_sptr = input_image_sptr->clone();		
      DiscretisedDensity<3,float>::full_iterator out_iter = output_image_sptr->begin_all();
      DiscretisedDensity<3,float>::const_full_iterator in_iter = input_image_sptr->begin_all_const();
      while( in_iter != input_image_sptr->end_all_const())
	{
	  if (*in_iter<0.F)
	    *out_iter = -(*in_iter);
	  ++in_iter; ++out_iter;
	}  
      Succeeded success =
	OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
	write_to_file(output_filename, *output_image_sptr);
      return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
    }
}
