//
// $Id$
//
/*
  Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
 
  \brief This programme attempts to get rid of the central artefact
  caused by normalisation+scatter problems on the 966. 

  \see cleanup966ImageProcessor

  \author Kris Thielemans
  \author Charalampos Tsoumpas [updates for the STIR release 2.0, but takes no credit for it. ]

  $Date$
  $Revision$
*/

#include "stir/DiscretisedDensity.h"
#include "local/stir/cleanup966ImageProcessor.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc!=3) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> \n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  // apply 
  cleanup966ImageProcessor<float> image_processor;
  if (image_processor.set_up(*density_ptr) == Succeeded::no)
    {
      warning("%s: cannot handle non-zero offset in x or y in file %s\n",
	      argv[0], input_filename);
      return EXIT_FAILURE;
    }
  image_processor.apply(*density_ptr);

  // write image
  const Succeeded writing_succeeded=OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->  
    write_to_file(output_filename,*density_ptr); 

  return writing_succeeded==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
