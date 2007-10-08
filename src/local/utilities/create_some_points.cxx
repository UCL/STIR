//
// $Id$
//
/*
  Copyright (C) 2007 - $Date$, Hammersmith Imanet Ltd
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
  \brief It produces some single points based on an image. 
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \todo Potentially, in the future make it more sophisticated.

  \par Usage:
  \code
  create_some_points output_image [1 | !1]
  \endcode
*/

#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR
 
int main(int argc, char *argv[])
{ 
  if (argc<2 || argc>3)
    {
     std::cerr << "Usage:" << argv[0] << " image [1 or else]\n";    
     return EXIT_FAILURE;
    }
  const int option = argc==3 ? atoi(argv[2]) : 1;
  shared_ptr< DiscretisedDensity<3,float> >  image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& image = *image_sptr;
  image.fill(0);
  if(option==1)
    {
      image[15][-20][0]=.1;
      image[25][20][0]=.1;
      image[35][20][20]=.1;
      image[45][20][-20]=.1;
      image[55][0][0]=.1;
    }
  else
    {
      image[10][0][0]=.1;
      image[20][0][-20]=.1;
      image[30][0][20]=.1;
      image[40][-20][-20]=.1;
      image[50][-20][20]=.1;
    }

  OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(argv[1], *image_sptr);
	
  return EXIT_SUCCESS;
}
  
