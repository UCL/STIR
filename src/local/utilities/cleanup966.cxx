//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme attempts to get rid of the central artefact
  caused by normalisation+scatter problems on the 966. 

  \see cleanup966ImageProcessor

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"
#include "local/stir/cleanup966ImageProcessor.h"
#include "stir/IO/DefaultOutputFileFormat.h"
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
  DefaultOutputFileFormat output_file_format;
  Succeeded res = 
    output_file_format.write_to_file(output_filename, *density_ptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}




