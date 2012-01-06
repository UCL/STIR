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

  \brief A preliminary utility to flip the x-axis of an image.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/DiscretisedDensity.h"
#include <algorithm>
#include <unistd.h>

int main(int argc, char * argv[])
{

  USING_NAMESPACE_STIR;
  const char * output_filename = 0;
  const char * input_filename = 0;

  const char * const usage = "image_flip_x -o output_filename -i input_filename\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt (argc, argv, "i:o:")) != -1)
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

  shared_ptr<DiscretisedDensity<3,float> > density_ptr
    (read_from_file<DiscretisedDensity<3,float> >(input_filename));

  for (DiscretisedDensity<3,float>::iterator z_iter = density_ptr->begin();
       z_iter != density_ptr->end();
       ++z_iter)
    {
      for (Array<2,float>::iterator y_iter = z_iter->begin();
	   y_iter != z_iter->end();
	   ++y_iter)
	{
	  for (int left=y_iter->get_min_index(), right=y_iter->get_max_index();
	       left < right;
	       ++left, --right)
	    std::swap((*y_iter)[left], (*y_iter)[right]);
	}
    }
  
  Succeeded success =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, *density_ptr);
  
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
