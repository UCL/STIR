//
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme finds plane rescaling factors

  It requires an input image where each plane total should be rescaled to 
  the same number (e.g. a reconstruction of a uniform cylinder). Scale factors
  are computed as 1/(plane_total/average_plane_total).

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2012, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"
#include "stir/stream.h"
#include "stir/Array.h"
#include "stir/IO/read_from_file.h"
#include "stir/error.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ofstream;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=3) {
    cerr<<"Usage: " << argv[0] << " <output filename> <input image filename>\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr 
    (read_from_file<DiscretisedDensity<3,float> >(input_filename));


  // store plane sums
  Array<1,float> rescale_factors(density_ptr->get_min_index(), density_ptr->get_max_index());
  for (int z=density_ptr->get_min_index(); 
       z<=density_ptr->get_max_index();
       ++z)
  {
    rescale_factors[z] = (*density_ptr)[z].sum();
  }

  // normalise to an average of 1
  const float average = rescale_factors.sum()/ rescale_factors.get_length();
  rescale_factors /= average;

  // invert
  for (int z=density_ptr->get_min_index(); 
       z<=density_ptr->get_max_index();
       ++z)
  {
    if (rescale_factors[z]<1E-4)
    {
      warning("%s: plane total for plane %d is less than 1E-4 the average.\n"
              "Setting scale factor to 1E4\n",
              argv[0], z-density_ptr->get_min_index()+1);
      rescale_factors[z] = 10000.F;
    }
    else
      rescale_factors[z] = 1/rescale_factors[z];
  }

  // write to file
  {
    ofstream factors(output_filename);
    factors << rescale_factors;
  }

  return EXIT_SUCCESS;

}




