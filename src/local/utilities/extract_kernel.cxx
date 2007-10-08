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
  extract_kernel kernel_name input_image z_location y_location x_location z_size y_size x_size 
  \endcode
*/

#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/Succeeded.h"


USING_NAMESPACE_STIR
 
int main(int argc, char *argv[])
{ 
  if (argc!=9)
    {
     std::cerr << "Usage:" << argv[0] 
	       << "  kernel_name input_image z_location y_location x_location z_size y_size x_size \n"
	       << "The kernel sizes: z_size y_size and x_size must be odd at the moment.\n" ; 
     return EXIT_FAILURE;
    }

  const int z_location=atoi(argv[3]);
  const int y_location=atoi(argv[4]);
  const int x_location=atoi(argv[5]);
  const int z_size = atoi(argv[6]);
  const int y_size = atoi(argv[7]);
  const int x_size = atoi(argv[8]);

  if (z_size%2==0 && y_size%2==0 && x_size%2==0)
    error("The kernel sizes: z_size y_size and x_size must be odd at the moment.\n");

  shared_ptr< DiscretisedDensity<3,float> >  kernel_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& kernel = *kernel_sptr;

  shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);

 const DiscretisedDensityOnCartesianGrid<3,float>*  input_image_cartesian_ptr = 
    dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (input_image_sptr.get());
  const DiscretisedDensityOnCartesianGrid<3,float>& input_image = *input_image_cartesian_ptr;

  const int z_min_input=input_image.get_min_index();
  const int z_max_input=input_image.get_max_index();  
  const int y_min_input=input_image[z_min_input].get_min_index();
  const int y_max_input=input_image[z_min_input].get_max_index();
  const int x_min_input=input_image[z_min_input][y_min_input].get_min_index();
  const int x_max_input=input_image[z_min_input][y_min_input].get_max_index();
  kernel.fill(0);

  for (int k=0 ; k <=z_size-1 ; ++k)
    {
      const int k_input=z_location+k-static_cast<int>(static_cast<float>(z_size-1)/2);
      for (int j=static_cast<int>((static_cast<float>(-y_size+1))/2.F) ; j<=static_cast<int>((static_cast<float>(y_size-1))/2.F) ; ++j)
	{
	  const int j_input=y_location+j;	  
	  for (int i=static_cast<int>((static_cast<float>(-x_size+1))/2.F) ; i<=static_cast<int>((static_cast<float>(x_size-1))/2.F) ; ++i)
	    {
	      const int i_input=x_location+i;
#if 0 
		  std::cerr << "For k=" << k << " from original k: " << k_input  << "\n" ;  
		  std::cerr << "For j=" << j << " from original j: " << j_input  << "\n" ;  
		  std::cerr << "For i=" << i << " from original i: " << i_input  << "\n\n" ;  	      
#endif
	      if( 
		 (k_input>=z_min_input) &&
		 (k_input<=z_max_input) && 
		 (j_input>=y_min_input) && 
		 (j_input<=y_max_input) && 
		 (i_input>=x_min_input) && 
		 (i_input<=x_max_input) )
		kernel[k][j][i]=input_image[k_input][j_input][i_input];
	    }
	}
    }
	  Succeeded success =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(argv[1], kernel);
	
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
 
