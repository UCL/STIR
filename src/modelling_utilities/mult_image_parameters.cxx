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
  \brief It multiplies the two parametric images and produces one-parameter image.
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  \par Usage:
  \code
  mult_image_parameters -o output_filename -i input_filename 
  \endcode

  \attention Assumes that only two parameters exist.
  \todo Make a generic utility which will multiply all the parameters together and store them in a multiple image file.
  \todo It might be possible to integrate it into the stir_math.cxx, in the future.

  \note It is useful to estimate the covariance between the two parameters of the parametric images. 

  $Date$
  $Revision$
*/
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include <algorithm>
#include <unistd.h>

int main(int argc, char * argv[])
{

  USING_NAMESPACE_STIR;
  const char * output_filename = 0;
  const char * input_filename = 0;

  const char * const usage = "mult_image_parameters -o output_filename -i input_filename\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt (argc, argv, "i:o:p")) != -1)
    switch (c)
      {
      case 'i':
	input_filename = optarg;
	break;
      case 'o':
	output_filename = optarg;
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

  const shared_ptr<ParametricVoxelsOnCartesianGrid> input_image_sptr = 
    ParametricVoxelsOnCartesianGrid::read_from_file(input_filename);  
  const ParametricVoxelsOnCartesianGrid & input_image = *input_image_sptr;  

  shared_ptr<DiscretisedDensity<3,float> > output_image_sptr = (input_image_sptr->construct_single_density(1)).clone();
  DiscretisedDensity<3,float>& output_image = *output_image_sptr;

  const int min_k_index = output_image.get_min_index(); 
  const int max_k_index = output_image.get_max_index();
  for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = output_image[k].get_min_index(); 
      const int max_j_index = output_image[k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
	{
	  const int min_i_index = output_image[k][j].get_min_index(); 
	  const int max_i_index = output_image[k][j].get_max_index();
	  for ( int i = min_i_index; i<= max_i_index; ++i)	    
	    output_image[k][j][i]=input_image[k][j][i][1]*input_image[k][j][i][2];
	}
    }
  Succeeded success =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, *output_image_sptr);

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}  

