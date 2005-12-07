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

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/TimeFrameDefinitions.h"
#include "local/stir/DynamicDiscretisedDensity.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/numerics/linear_integral.h"
#include "stir/linear_regression.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/Succeeded.h"
#include "stir/VectorWithOffset.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>
//#include <cstring>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
using std::setw;
#endif

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
  if (argc<3 || argc>8)
    {
      std::cerr << "Usage:" << argv[0] << "\n"
	   << "\t[dynamic_image_filename]\n"
	   << "\t[plasma_data_filename]\n" 
	   << "\t[input function time_shift in sec]\n" 
	   << "\t[blood volume (bv)]\n" 
	   << "\t[Model starting time]\n" 
	   << "\t[is_decay_corrected]\n"
	   << "\ttime_shift: default to 13 sec\n"
	   << "\tbv: default to 0.05\n" 
	   << "\tstarting_time: default to 0 sec\n"
	   << "\tis_decay_corrected: is set to false. Give 1 for true.\n";
      return EXIT_FAILURE;            
    }       
  const float time_shift = argc>=4 ? atof(argv[3]) : 13.F ;
  const float bv = argc>=5 ? atof(argv[4]) : 0.05F ; // Blood Volume, usually constant. ChT::ToDo:Maybe add to the PlasmaData class!!!
  const float starting_time = argc>=6 ? atof(argv[5]) : 0.F ;
  const bool is_decay_corrected = argc>=7 ? atoi(argv[6]) : false ;
  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
  DynamicDiscretisedDensity::read_from_file(argv[1]);
  DynamicDiscretisedDensity dyn_image = *dyn_image_sptr;// NOT REFERENCE BECAUSE THEY ARE CORRECTED AFTERWARDS!!!

  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  string input_string(argv[1]);
  // Prepare slope_image
  string slope_string;
  slope_string += "slope_it_"+ input_string  ;
  shared_ptr<DiscretisedDensity<3,float> > slope_image_sptr = 
    dyn_image_sptr->get_density(1).clone();
  DiscretisedDensity<3,float>& slope_image = *slope_image_sptr;
  // Prepare constant_image
  string constant_string;
  constant_string += "constant_it_"+ input_string  ;
  shared_ptr<DiscretisedDensity<3,float> > constant_image_sptr = 
    dyn_image_sptr->get_density(1).clone();
  DiscretisedDensity<3,float>& constant_image = *constant_image_sptr;

  // Apply calibration_factor and Decay Correct Frame Image Data
   dyn_image.calibrate_frames();
   if(!is_decay_corrected)
     dyn_image.decay_correct_frames();

  // Read the Plasma Data from the given file.
  // Shift the Plasma Data as appropriately. 
  PlasmaData plasma_data;
  plasma_data.read_plasma_data(argv[2]);   // The implementation currently assumes three list file of blood. 
  plasma_data.shift_time(time_shift);

  // Initialise the vector where the plasma values for each frame to 0. 
  std::vector<float>  plasma_vector(num_frames,0) ; 
  std::vector<float>  blood_vector(num_frames,0) ; 
  std::vector<float>  plasma_sum_vector(num_frames,0) ; //I use sum instead of mean, since PET frames estimate the sum.
   
  PlasmaData::const_iterator cur_iter;
  unsigned int frame_num;
  //Short plasma data in frames
  // Estimate the plasma_frame_vector and the plasma_frame_sum_vector using th linear_integral() implementation
  for (  frame_num = 1;  //ChT::TakeCare: Frames time is assumed not to overlap.
	 frame_num<=num_frames ; ++frame_num )
    {     
      std::vector<float> time_frame_vector ; 
      std::vector<float> plasma_frame_vector ;
      std::vector<float> blood_frame_vector ;

      for( cur_iter=plasma_data.begin() ; cur_iter!=plasma_data.end(); ++cur_iter)
    	{
	  const float cur_time=(*cur_iter).get_time_in_s() ;
	  if ( cur_time<(dyn_image.get_time_frame_definitions()).get_start_time(frame_num))
	    continue;

	  const float cur_plasma_cnt=(*cur_iter).get_plasma_counts_in_kBq();
	  const float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	  if ( cur_time<(dyn_image.get_time_frame_definitions()).get_end_time(frame_num))
	    {
	      plasma_frame_vector.push_back(cur_plasma_cnt);
	      blood_frame_vector.push_back(cur_blood_cnt);
	      time_frame_vector.push_back(cur_time);
	      //      float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	    }
	  else
	    {
	      plasma_vector[frame_num-1]=linear_integral(plasma_frame_vector,time_frame_vector) ;
	      blood_vector[frame_num-1]=linear_integral(blood_frame_vector,time_frame_vector) ;
	      //	      std::cerr << cur_time << " " << plasma_vector[frame_num-1] << "\n";
	      break;
	    }
	}
	    plasma_sum_vector[frame_num-1]+=plasma_vector[frame_num-1] ;
	    // The reconstructed CTI images and the STIR (when using reconstruction script) are always devided by the time frame duration.
	    plasma_vector[frame_num-1]/=(dyn_image.get_time_frame_definitions()).get_duration(frame_num);
    }

  // Do linear_regression for each voxel 
  // for k j i 
  {
    float scale=0.F;
    float constant=0.F;
    float variance_of_scale=0.F;
    float variance_of_constant=0.F;
    float covariance_of_constant_with_scale=0.F;
    float chi_square = 0.F;  
  
    VectorWithOffset<float> patlak_y(0,num_frames-1);
    VectorWithOffset<float> patlak_x(0,num_frames-1);
    VectorWithOffset<float> weights(0,num_frames-1);
#if 0 // This is to simplify the method, but not working, yet.
 
	 DiscretisedDensity<3,float>::full_iterator slope_iter = slope_image.begin_all();
	 while( slope_iter != slope_image.end_all()) //ChT::ToDo:ParametricImage Class
		{
      		  const int counter=slope_iter-slope_image.begin_all();
		  for (  frame_num = 1;  //ChT::TakeCare: Frames time is assumed not to overlap.
			   frame_num<=num_frames ; ++frame_num )
		    {
		        DiscretisedDensity<3,float>::const_full_iterator frame_iter = dyn_image[frame_num].begin_all_const();
       			patlak_y[frame_num-1]=*(counter+frame_iter)-bv*blood_vector[frame_num-1])/plasma_vector[frame_num-1];
			patlak_x[frame_num-1]=plasma_sum_vector[frame_num-1]/plasma_vector[frame_num-1];
			weights[frame_num-1]=1;
			linear_regression(constant, scale,
				       chi_square,
				       variance_of_constant,
				       variance_of_scale,
				       covariance_of_constant_with_scale,
				       patlak_y,
				       patlak_x,		      
				       weights);
			*slope_iter=scale;
		}
	       ++slope_iter;
       }
#else
	const int min_k_index = slope_image.get_min_index(); 
	const int max_k_index = slope_image.get_max_index();
	for ( int k = min_k_index; k<= max_k_index; ++k)
	  {
	    const int min_j_index = slope_image[k].get_min_index(); 
	    const int max_j_index = slope_image[k].get_max_index();
	    for ( int j = min_j_index; j<= max_j_index; ++j)
	      {
		const int min_i_index = slope_image[k][j].get_min_index(); 
		const int max_i_index = slope_image[k][j].get_max_index();
		for ( int i = min_i_index; i<= max_i_index; ++i)
		   {
		     for (  frame_num = 1;  //ChT::TakeCare: Frames time is assumed not to overlap.
			   frame_num<=num_frames ; ++frame_num )
		      {
			patlak_y[frame_num-1]=(dyn_image[frame_num][k][j][i]
					       -bv*blood_vector[frame_num-1])/plasma_vector[frame_num-1];
			patlak_x[frame_num-1]=plasma_sum_vector[frame_num-1]/plasma_vector[frame_num-1];
			weights[frame_num-1]=1;
		      }
		     linear_regression(constant, scale,
				       chi_square,
				       variance_of_constant,
				       variance_of_scale,
				       covariance_of_constant_with_scale,
				       patlak_y,
				       patlak_x,		      
				       weights);			
		     slope_image[k][j][i]=scale;
		     constant_image[k][j][i]=constant;
	      }
	  }
	  }
#endif
  }

  shared_ptr<OutputFileFormat> output_format_sptr =
    new DefaultOutputFileFormat;
  std::cerr << "Writing 'slope'-image: " << slope_string << endl;

  Succeeded success =
    output_format_sptr->write_to_file(slope_string, *slope_image_sptr);
  std::cerr << "Writing 'constant'-image: " << constant_string << endl;
  success = Succeeded::yes ? output_format_sptr->write_to_file(constant_string, *constant_image_sptr) : Succeeded::no ;
  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;	
}


