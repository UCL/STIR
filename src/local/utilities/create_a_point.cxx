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
  create_a_point output_image_name image_template z_location y_location x_location value // 
  \endcode
*/

#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR
 
int main(int argc, char *argv[])
{ 
  if (argc<6 || argc>7)
    {
     std::cerr << "Usage:" << argv[0] 
	       << "output_image_name image_template z_voxel_location y_voxel_location x_voxel_location value \n";       
     return EXIT_FAILURE;
    }

  const int z_location=atoi(argv[3]);
  const int y_location=atoi(argv[4]);
  const int x_location=atoi(argv[5]);
  const float value = argc>=7 ? atof(argv[6]) : 1.F;

  shared_ptr< DiscretisedDensity<3,float> >  image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);
  DiscretisedDensity<3,float>& output_image = *image_sptr;
  output_image.fill(0);
  output_image[z_location][y_location][x_location]=value;
  Succeeded success =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(argv[1], output_image);
	
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}
 
