//
// $Id$
//

/*!
  \file
  \ingroup utilities 
  \brief simple multiplying of images

  \author Kris Thielemans

  $Date$
  $Revision$

  \warning This does not check if image types are the same, nor if origins and
  voxel sizes etc. match.
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/DiscretisedDensity.h"
#include "stir/interfile.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
#endif



//********************** main



USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{


  if(argc<4)
  {
    cerr<< "Usage: " << argv[0] << " out_image image1 image2 [image3...]\n";
    exit(EXIT_FAILURE);
  }

  --argc;
  ++argv;
  const char * const out_image_filename = *argv;

  --argc;
  ++argv;
  cout << "Reading image " << *argv << endl;
  shared_ptr< DiscretisedDensity<3,float> >  image_ptr = 
    DiscretisedDensity<3,float>::read_from_file(*argv);


  while(--argc > 0)
  {
    ++argv;
    cout << "Reading image " << *argv << endl;
    shared_ptr< DiscretisedDensity<3,float> >  image_to_multiply_ptr = 
    DiscretisedDensity<3,float>::read_from_file(*argv);

    *image_ptr *= *image_to_multiply_ptr;
  }

  cout << "Writing multiply_image " << out_image_filename << endl;
  write_basic_interfile(out_image_filename, *image_ptr);

  return EXIT_SUCCESS;

} //end main
