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
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/modulo.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc<3 || argc>5) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> [spacing [size]]\n"
	<< "spacing defaults to 2, size to 1";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  const int spacing = argc>3?atoi(argv[3]):2;
  const int size = argc>4?atoi(argv[4]):1;
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_sptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  BasicCoordinate<3,int> c;
  const int min1=density_sptr->get_min_index();
  const int max1=density_sptr->get_max_index();
  for (c[1]=min1; c[1]<=max1; ++c[1])
    {
      const int min2=(*density_sptr)[c[1]].get_min_index();
      const int max2=(*density_sptr)[c[1]].get_max_index();

      for (c[2]=min2; c[2]<=max2; ++c[2])
	{
	  const int min3=(*density_sptr)[c[1]][c[2]].get_min_index();
	  const int max3=(*density_sptr)[c[1]][c[2]].get_max_index();
	  for (c[3]=min3; c[3]<=max3; ++c[3])
	    {
	      if (modulo(c[2],spacing)<size && modulo(c[3],spacing)<size)
		(*density_sptr)[c] = 1;
	      else
		(*density_sptr)[c] = 0;
	    }
	}
    }
  

  // write image
  DefaultOutputFileFormat output_file_format;
  Succeeded res = 
    output_file_format.write_to_file(output_filename, *density_sptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}




