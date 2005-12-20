
//
// $id: apply_patlak_on_images.cxx,v 1.1 2005/12/02 16:22:23 ctsoumpas Exp $
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

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/TimeFrameDefinitions.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/modelling/PlasmaSample.h"
#include "local/stir/modelling/BloodFrame.h"
#include "local/stir/modelling/BloodFrameData.h"
#include "local/stir/modelling/patlak.h"
#include "local/stir/numerics/linear_integral.h"
#include "stir/linear_regression.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/Succeeded.h"
#include "stir/VectorWithOffset.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::ifstream;
using std::istream;
using std::setw;
#endif

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
    if (argc<3 || argc>9)
      {
	std::cerr << "Usage:" << argv[0] << "\n"
		  << "\t[dynamic_image_filename]\n"
		  << "\t[plasma_data_filename]\n" 
		  << "\t[input function time_shift in sec]\n" 
		  << "\t[blood volume (bv)]\n" 
		  << "\t[starting]\n" 
		  << "\t[is_decay_corrected]\n"
		  << "\t[is_calibrated]\n"
		  << "\t[if_image_based]\n"
		  << "\ttime_shift: default to 13 sec\n"
		  << "\tbv: default to 0.05\n" 
		  << "\tstarting_frame: default 13 to take the last 13 frames.\n"
		  << "\tis_decay_corrected: is set to false for both PlasmaData and DynamicDiscretisedDensity. \n"
  		  << "\tis_calibrated: is set to true. False will cause an error. \n"
  		  << "\tInput Function (if) Image-Based: is set to false. Give 1 for true. \n";
	return EXIT_FAILURE;            
      }       
  const float time_shift = argc>=4 ? atof(argv[3]) : 13.F ;
  const bool if_image_based = argc>=9 ? atoi(argv[8]) : false ;
  const float bv = argc>=5 ? (if_image_based ? 0 : atof(argv[4])) : 0.05F ; // Blood Volume, usually constant. ChT::ToDo:Maybe add to the PlasmaData class!!! //ChT::Check: If this works as I expect.

  const bool is_decay_corrected = argc>=7 ? atoi(argv[6]) : false ;
  const bool is_calibrated = argc>=8 ? atoi(argv[7]) : true ;

  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
    DynamicDiscretisedDensity::read_from_file(argv[1]);
  DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;
  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  if (num_frames<=13)
    error("Current Patlak implementation default takes the last ${starting_frame} frames. /nGive less than ${starting_frame} but be aware of the plot!");
  const unsigned int starting_frame = argc>=6 ? atoi(argv[5]) : num_frames-13+1 ;

  string input_string(argv[1]);
  replace_extension(input_string, "");
  // Prepare slope_image
  string slope_string;
  slope_string = "slope_"+ input_string  ;
  shared_ptr<DiscretisedDensity<3,float> > slope_image_sptr = 
    dyn_image_sptr->get_density(1).clone();
  DiscretisedDensity<3,float>& slope_image = *slope_image_sptr;
  // Prepare y_intersection_image
  string y_intersection_string;
  y_intersection_string = "y_intersection_"+ input_string  ;
  shared_ptr<DiscretisedDensity<3,float> > y_intersection_image_sptr = 
    dyn_image_sptr->get_density(1).clone();
  DiscretisedDensity<3,float>& y_intersection_image = *y_intersection_image_sptr;

  // Read the Plasma Data from the given file.
  // Shift the Plasma Data as appropriately. 
  PlasmaData plasma_data;
  if(!if_image_based)
    {     
      plasma_data.read_plasma_data(argv[2]);   // The implementation assumes three list file of blood. 
      plasma_data.shift_time(time_shift);
    apply_patlak_to_images_and_arterial_sampling(y_intersection_image, 
						 slope_image, 
						 dyn_image,  
						 plasma_data, 
						 time_shift, 
						 starting_frame, 
						 bv, 
						 is_calibrated, 
						 is_decay_corrected);

    }
  if(if_image_based)
  {
    BloodFrameData blood_frame_data_temp; //ChT::const_iterator I Think was the problem to set_time_in_s(frame_time);
    blood_frame_data_temp.read_blood_frame_data(argv[2]);   // The implementation assumes two list file. 
    std::vector<BloodFrame> blood_plot;
    for( BloodFrameData::const_iterator cur_iter=blood_frame_data_temp.begin() ; cur_iter!=blood_frame_data_temp.end() ; ++cur_iter)
      {
	const unsigned int cur_frame=(*cur_iter).get_frame_num() ;
	const float mean_time=(dyn_image.get_time_frame_definitions()).get_start_time(cur_frame)//ChT::ThinkAgain
	  +(0.5F*(dyn_image.get_time_frame_definitions()).get_duration(cur_frame));
	const BloodFrame blood_frame(cur_frame,mean_time,cur_iter->get_blood_counts_in_kBq());	  
	blood_plot.push_back(blood_frame); 
	cerr <<  mean_time <<" "; 
      }
    BloodFrameData blood_frame_data(blood_plot);
    apply_patlak_to_images_plasma_based(y_intersection_image,
					slope_image, 
					dyn_image,
					blood_frame_data,
					starting_frame, 
					is_calibrated , 
					is_decay_corrected);
  }

  // Writing images to file

  std::cerr << "Writing 'y_intersection'-image: " << y_intersection_string << "\n";
  Succeeded slope_success =
    write_basic_interfile(y_intersection_string, *y_intersection_image_sptr);
  std::cerr << "Writing 'slope'-image: " << slope_string << "\n";  
  Succeeded y_intersection_success = 
    write_basic_interfile(slope_string, *slope_image_sptr);
  if (y_intersection_success==Succeeded::yes && slope_success==Succeeded::yes)
    return EXIT_SUCCESS ;
  else 
    return EXIT_FAILURE ;	
}
