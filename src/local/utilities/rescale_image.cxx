//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme rescales an image with a global scaling factor given ion the command line
  
  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"
#include "stir/interfile.h"
#include "stir/Succeeded.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc!=4) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> <rescale factor (float)>\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  const float rescale_factor = static_cast<float>(atof(argv[3]));
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  // apply scale factor
  (*density_ptr) *= rescale_factor;
      
  // write image
  Succeeded res = write_basic_interfile(output_filename, *density_ptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}




