//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief List coordinates of maxima in the image to stdout

  If more than 1 maximum is asked for, repeat the following:

  # find the current maximum, and then
  # mask the neighbourhood (specified as a box) out
  # go to step 1

  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$

  \par Usage:
   \code
   find_maxima_in_image filename [num_maxima [ half_mask_size_xy [half_mask_size z]] ]
   \endcode
   \param num_maxima defaults to 1
   \param half_mask_size_xy defaults to 1 (giving a 3x3 mask in the plane)
   \param half_mask_size_z defaults to \a half_mask_size_xy
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"

#include <iostream>
#include <iomanip>
#include <algorithm>


#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
using std::setw;
#endif

USING_NAMESPACE_STIR
 
int main(int argc, char *argv[])
{ 
  
  if (argc< 2 || argc>5)
  {
    cerr << "Usage:" << argv[0] << " input_image [num_maxima [ half_mask_size_xy [half_mask_size z]] ]\n"
	 << "\tnum_maxima defaults to 1\n"
	 << "\thalf_mask_size_xy defaults to 1\n"
	 << "\thalf_mask_size_z defaults to half_mask_size_xy" <<endl;    
    return (EXIT_FAILURE);
  }

  const unsigned num_maxima = argc>=3 ? atoi(argv[2]) : 1;
  const int mask_size_xy = argc>=4 ? atoi(argv[3]) : 1;
  const int mask_size_z = argc>=5 ? atoi(argv[4]) : mask_size_xy;

  cerr << "Finding " << num_maxima << " maxima, each at least \n\t"
       << mask_size_xy << " pixels from each other in x,y direction, and\n\t"
       << mask_size_z << " pixels from each other in z direction.\n";

  shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& input_image = *input_image_sptr;

  const float mask_value = std::min(input_image.find_min()-100, -1.E20F);

  for (unsigned int maximum_num=0; maximum_num!=num_maxima; ++ maximum_num)
    {
      const float current_maximum = input_image.find_max();
      int max_k=0, max_j=0,max_i=0; // initialise to avoid compiler warnings
      bool found=false;
      {
	const int min_k_index = input_image.get_min_index(); 
	const int max_k_index = input_image.get_max_index();
	for ( int k = min_k_index; k<= max_k_index && !found; ++k)
	  {
	    const int min_j_index = input_image[k].get_min_index(); 
	    const int max_j_index = input_image[k].get_max_index();
	    for ( int j = min_j_index; j<= max_j_index && !found; ++j)
	      {
		const int min_i_index = input_image[k][j].get_min_index(); 
		const int max_i_index = input_image[k][j].get_max_index();
		for ( int i = min_i_index; i<= max_i_index && !found; ++i)
		  {
		    if ( input_image[k][j][i] == current_maximum)
		      {
			max_k = k;
			max_j = j;
			max_i = i;
			cout << "max: " << setw(6) << current_maximum 
			     << " at: " 
			     << setw(3) << max_k 
			     << ',' << setw(3) << max_j 
			     << ',' << setw(3)  << max_i;
			  {
			    BasicCoordinate<3,float> phys_coord = 
			      input_image.
			      get_physical_coordinates_for_indices(make_coordinate(max_k, max_j, max_i));
			    cout << " which is "
				 << setw(6) << phys_coord[1] 
				 << ',' << setw(6) << phys_coord[2] 
				 << ',' << setw(6)  <<phys_coord[3]
				 << " in mm in physical coordinates";
			  }
			cout << '\n';
			found = true;		  
		      }
		  }
	      }
	  }
	if (!found)
	  {
	    warning("Something strange going on: can't find maximum %g\n", current_maximum);
	    return EXIT_FAILURE;
	  }
	if (maximum_num+1!=num_maxima)
	  {
	    // now mask it out for next run
	    for ( int k = std::max(max_k-mask_size_z,min_k_index); k<= std::min(max_k+mask_size_z,max_k_index); ++k)
	    {
	      const int min_j_index = input_image[k].get_min_index(); 
	      const int max_j_index = input_image[k].get_max_index();
	      for ( int j = std::max(max_j-mask_size_xy,min_j_index); j<= std::min(max_j+mask_size_xy,max_j_index); ++j)
	      {
		const int min_i_index = input_image[k][j].get_min_index(); 
		const int max_i_index = input_image[k][j].get_max_index();
		for ( int i = std::max(max_i-mask_size_xy,min_i_index); i<= std::min(max_i+mask_size_xy,max_i_index); ++i)
		  input_image[k][j][i] = mask_value;
	      }
	    }
	  }
      }
    }
  return EXIT_SUCCESS;
}
  
