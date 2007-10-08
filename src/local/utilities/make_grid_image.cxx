//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief This program creates a simple grid image
  \author Kris Thielemans
  $Date$
  $Revision$
*/
#include "stir/DiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/modulo.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc<3 || argc>7) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> [xy-spacing [xy-size][z-spacing [z-size] ] ] ]\n"
	<< "xy-spacing defaults to 2, xy-size to 1\n"
	<< "z-spacing defaults to 1, z-size to 1\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  const int xy_spacing = argc>3?atoi(argv[3]):2;
  const int xy_size = argc>4?atoi(argv[4]):1;
  const int z_spacing = argc>5?atoi(argv[5]):1;
  const int z_size = argc>6?atoi(argv[6]):1;
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

      if (modulo(c[1],z_spacing)>z_size)
	continue; // just zeroes on this plane

      for (c[2]=min2; c[2]<=max2; ++c[2])
	{
	  const int min3=(*density_sptr)[c[1]][c[2]].get_min_index();
	  const int max3=(*density_sptr)[c[1]][c[2]].get_max_index();
	  for (c[3]=min3; c[3]<=max3; ++c[3])
	    {
	      if (modulo(c[2],xy_spacing)<xy_size && modulo(c[3],xy_spacing)<xy_size)
		(*density_sptr)[c] = 1;
	      else
		(*density_sptr)[c] = 0;
	    }
	}
    }
  

  // write image
  Succeeded res = 
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, *density_sptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}




