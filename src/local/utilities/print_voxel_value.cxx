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
  \brief It prints the value of the requested voxel of an image. 
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \todo Potentially, in the future make it more sophisticated.

  \par Usage:
  \code
  print_voxel_value image z_location y_location x_location // 
  \endcode
*/

#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR
 
int main(int argc, char *argv[])
{ 
  if (argc!=5)
    {
     std::cerr << "Usage: " << argv[0] 
	       << "image z_voxel_location y_voxel_location x_voxel_location \n";       
     return EXIT_FAILURE;
    }

  const int z_location=atoi(argv[2]);
  const int y_location=atoi(argv[3]);
  const int x_location=atoi(argv[4]);

  const shared_ptr< DiscretisedDensity<3,float> >  image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);
  std::cout << (*image_sptr)[z_location][y_location][x_location] << endl;

	
  return EXIT_SUCCESS;

}
 
