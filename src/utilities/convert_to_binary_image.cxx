//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
 
  \brief This program outputs a binary image (voxel values 0 or 1) where all values 
  strictly below a threshold are set to 0, and others to 1.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=4) {
    std::cerr<<"Usage: " << argv[0] << " <output filename> <input filename> threshold\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  std::string output_filename = argv[1];
  char const * const input_filename = argv[2];
  const float threshold = atof(argv[3]);
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  
    density_ptr(read_from_file<DiscretisedDensity<3,float> >(input_filename));

  // binarise
  for (DiscretisedDensity<3,float>::full_iterator iter = density_ptr->begin_all();
       iter != density_ptr->end_all();
	   ++iter)
	{
	  if (*iter < threshold)
		     *iter = 0.F;
	  else
		*iter = 1.F;
	}
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
  output_file_format_sptr->set_type_of_numbers(NumericType::SCHAR);
  output_file_format_sptr->set_scale_to_write_data(1);

  return 
    output_file_format_sptr->write_to_file(output_filename, *density_ptr) == Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;

}
