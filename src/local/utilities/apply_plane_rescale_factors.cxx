//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme rescales planes of an image according to a file with
  scale factors

  The scale factors should be given in a file as {plane1, plane2,...}, as written
  by compute_plane_rescale_factors. Planes are multiplied with the relevant scale factor.

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
#include "stir/stream.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc!=4) {
    cerr<<"Usage: " << argv[0] << " <output filename> <input image filename> <rescale factors filename>\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  char const * const rescale_factors_filename = argv[3];
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  // read factors
  VectorWithOffset<float> rescale_factors;
  {
    ifstream factors(rescale_factors_filename);
    factors >> rescale_factors;
    if (rescale_factors.get_length() != density_ptr->get_length())
      error("%s: wrong number of scale factors (%d) found in file %s, should be %d\n",
            argv[0], rescale_factors.get_length(), 
            rescale_factors_filename, density_ptr->get_length());
  }

  rescale_factors.set_offset(density_ptr->get_min_index());


  // apply them
  for (int z=density_ptr->get_min_index(); 
       z<=density_ptr->get_max_index();
       ++z)
  {
    (*density_ptr)[z] *= rescale_factors[z];
  }
      

  Succeeded res = write_basic_interfile(output_filename, *density_ptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}




