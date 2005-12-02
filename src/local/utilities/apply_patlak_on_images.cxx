//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Apply the Patlak linear fit using Dynamic Images
  \author Charalampos Tsoumpas
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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

#include "local/stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/TimeFrameDefinitions.h"
#include "local/stir/modelling/PlasmaData.h"

#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>
//#include <cstring>

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
  if (argc<2 || argc>3)
    {
      std::cerr << "Usage:" << argv[0] << "\n"
	   << "\t[dynamic_image_filename]\n"
	   << "\t[plasma_data_filename]\n" ;
      return EXIT_FAILURE;            
    }       

  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
  DynamicDiscretisedDensity::read_from_file(argv[1]);
  const DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;

  unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();

  // Read the Plasma Data from the given file.
  // Shift the Plasma Data as appropriately. 
  // ChT::ToDo:In the future this value should be either given externally, or estimated somehow, in the fit process.  
  PlasmaData plasma_data;
  plasma_data.read_plasma_data(argv[2]);   // The implementation currently assumes three list file of blood. 
  plasma_data.shift_time(13.F);

  // Initialise the vector where the plasma values for each frame will be stored a sub-vector. 
  vector<float>  plasma_vector(num_frames) ; 
  vector<float>  plasma_sum_vector(num_frames) ; //I use sum instead of mean, since PET frames estimate the sum.
  //  Initialised to 0 the above
  
  PlasmaData::const_iterator cur_iter;
  
  //Short plasma data in frames
  // Estimate the plasma_frame_vector and the plasma_frame_sum_vector using th linear_integral() implementation
  for (  cur_iter=plasma_data.begin(), unsigned int frame_num = 1 ;  //ChT::TakeCare: Frame time is assumed not to overlap.
	 cur_iter!=plasma_data.end(), frame_num<=num_frames ; ++frame_num, ++cur_iter )
    {
      vector<float> time_frame_vector ; 
      vector<float> plasma_frame_vector ;
    	{
	  float cur_time=(*cur_iter).get_time_in_s() ;
	  float cur_plasma_cnt=(*cur_iter).get_plasma_counts_in_kBq();
  //      float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	  if ( cur_time<(dyn_image.get_time_frame_definitions()).get_end_time(frame_num))
	    {
	      plasma_frame_vector.push_back(cur_plasma_cnt);
	      time_frame_vector.push_back(cur_time);
	      //      float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	    }
	  else
	    plasma_vector[frame_num]=linear_integral(plasma_frame_vector,time_frame_vector) ;
	}
	    plasma_sum_vector[frame_num]+=plasma_vector ;
    }
  ////////CERR RESULTS////////


  return EXIT_SUCCESS;
}

