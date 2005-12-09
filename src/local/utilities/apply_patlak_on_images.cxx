
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
#include "local/stir/DynamicDiscretisedDensity.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/numerics/linear_integral.h"
#include "stir/linear_regression.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/Succeeded.h"
#include "stir/VectorWithOffset.h"
#include "stir/IO/interfile.h"
//#include "stir/IO/DefaultOutputFileFormat.h"
#include <utility>
#include <vector>
#include <string>
#include <iostream>
//#include <fstream>
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
	   << "\tstarting_frame: default to 8 \n"
	   << "\tis_decay_corrected: is set to false. Give 1 for true.\n";
      return EXIT_FAILURE;            
    }       
  const float time_shift = argc>=4 ? atof(argv[3]) : 13.F ;
  const float bv = argc>=5 ? atof(argv[4]) : 0.05F ; // Blood Volume, usually constant. ChT::ToDo:Maybe add to the PlasmaData class!!!
  const unsigned int starting_frame = argc>=6 ? atoi(argv[5]) : 8 ;
  const bool is_decay_corrected = argc>=7 ? atoi(argv[6]) : false ;
  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
  DynamicDiscretisedDensity::read_from_file(argv[1]);
  DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;// NOT REFERENCE BECAUSE THEY ARE CORRECTED AFTERWARDS!!!

  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  string input_string(argv[1]);
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
  float plasma_frame_sum; //used to store the previous frame values
  PlasmaData::const_iterator cur_iter;
  unsigned int frame_num;
  //Short plasma data in frames
  // Estimate the plasma_frame_vector and the plasma_frame_sum_vector using th linear_integral() implementation
  for (  frame_num = 1; 
	 frame_num<=num_frames ; ++frame_num )
    {     
      std::vector<float> time_frame_vector ; 
      std::vector<float> plasma_frame_vector ;
      std::vector<float> blood_frame_vector ;

      for( cur_iter=plasma_data.begin() ; cur_iter!=plasma_data.end(); ++cur_iter)
    	{
	  const float frame_start_time=(dyn_image.get_time_frame_definitions()).get_start_time(frame_num);
	  const float frame_end_time=(dyn_image.get_time_frame_definitions()).get_end_time(frame_num);
	  const float cur_time=(*cur_iter).get_time_in_s() ;

	  if ( cur_time<frame_start_time)
	     continue;

	  const float cur_plasma_cnt=(*cur_iter).get_plasma_counts_in_kBq();
	  const float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	  if ( cur_time<frame_end_time)
	    {
	      //Create border value using linear interpolation: y=y1+(y2-y1)*(t-t1)/(t2-t1)
	      plasma_frame_vector.push_back(cur_plasma_cnt);
	      blood_frame_vector.push_back(cur_blood_cnt);
	      time_frame_vector.push_back(cur_time);	    
	    }
	  else
	      continue;
	      //Create border value using linear interpolation: y=y1+(y2-y1)*(t-t1)/(t2-t1)
	}
      plasma_vector[frame_num-1]=linear_integral(plasma_frame_vector,time_frame_vector) ;
      blood_vector[frame_num-1]=linear_integral(blood_frame_vector,time_frame_vector) ;
      std::cerr << "Mean: " << plasma_vector[frame_num-1] << "   \n";
      plasma_frame_sum+=plasma_vector[frame_num-1];
      plasma_sum_vector[frame_num-1]=plasma_frame_sum ;
      std::cerr << "Sum: " << plasma_sum_vector[frame_num-1] << "   \n";
  // The reconstructed CTI images and the STIR (when using reconstruction script) are always devided by the time frame duration.
      plasma_vector[frame_num-1]/=(dyn_image.get_time_frame_definitions()).get_duration(frame_num);
      blood_vector[frame_num-1]/=(dyn_image.get_time_frame_definitions()).get_duration(frame_num);
    }
  std::vector<float> patlak_y_RoI(num_frames);
  std::vector<float> patlak_x_RoI(num_frames);

  // Do linear_regression for each voxel 
  // for k j i 
  {
    float slope=0.F;
    float y_intersection=0.F;
    float variance_of_slope=0.F;
    float variance_of_y_intersection=0.F;
    float covariance_of_y_intersection_with_slope=0.F;
    float chi_square = 0.F;  
     
#if 0 // This is to simplify the method, but not working, yet.
 
	 DiscretisedDensity<3,float>::full_iterator slope_iter = slope_image.begin_all();
	 while( slope_iter != slope_image.end_all()) //ChT::ToDo:ParametricImage Class
		{
      		  const int counter=slope_iter-slope_image.begin_all();
		  for (  frame_num = 1;
			   frame_num<=num_frames ; ++frame_num )
		    {
		        DiscretisedDensity<3,float>::const_full_iterator frame_iter = dyn_image[frame_num].begin_all_const();
       			patlak_y[frame_num-1]=*(counter+frame_iter)-bv*blood_vector[frame_num-1])/plasma_vector[frame_num-1];
			patlak_x[frame_num-1]=plasma_sum_vector[frame_num-1]/plasma_vector[frame_num-1];
			weights[frame_num-1]=1;
			linear_regression(y_intersection, slope,
				       chi_square,
				       variance_of_y_intersection,
				       variance_of_slope,
				       covariance_of_y_intersection_with_slope,
				       patlak_y,
				       patlak_x,		      
				       weights);
			*slope_iter=slope;
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
		     VectorWithOffset<float> patlak_y(starting_frame-1,num_frames-1);
		     VectorWithOffset<float> patlak_x(starting_frame-1,num_frames-1);
		     VectorWithOffset<float> weights(starting_frame-1,num_frames-1);
		     for ( frame_num = starting_frame; // Instead of shifting time
			   frame_num<=num_frames ; ++frame_num )
		      {
			patlak_y[frame_num-1]=(dyn_image[frame_num][k][j][i]
					       -bv*blood_vector[frame_num-1])/plasma_vector[frame_num-1];
			patlak_x[frame_num-1]=plasma_sum_vector[frame_num-1]/plasma_vector[frame_num-1];
			weights[frame_num-1]=1;			
			
			if(abs(max_k_index+min_k_index-2*k)<=6 && 
			      abs(max_j_index+min_j_index-2*j)<=6 && 
			      abs(max_i_index+min_i_index-2*i)<=6)
			  {
			    //	    std::cerr << "location selected [" << k << "] ["  << j << "] ["  << i << "]\n " ;
			    patlak_y_RoI[frame_num-1]+=patlak_y[frame_num-1];
			    patlak_x_RoI[frame_num-1]+=patlak_x[frame_num-1];
			  }
		      }
		    
		     linear_regression(y_intersection, slope,
				       chi_square,
				       variance_of_y_intersection,
				       variance_of_slope,
				       covariance_of_y_intersection_with_slope,
				       patlak_y,
				       patlak_x,		      
				       weights);			
		     slope_image[k][j][i]=slope;
		     y_intersection_image[k][j][i]=y_intersection;
	      }
	  }
      }    
#endif
  }
{	
  // Writing TAC to file.
  std::cerr << "Testing TAC. Look at the text files!\n";	  
  string output_string("TAC.txt");	 
  std::ofstream out(output_string.c_str()); //output file //
  if(!out)
    std::cout << "Cannot open text file.\n" ; 
  out << "Frame" << "\tTimePoint" << "\t\tRoI-X" << "\t\tRoI-Y\n" ;

 for (  frame_num = starting_frame ; frame_num<=num_frames ; ++frame_num )
   out  << frame_num << "\t"
	<< (dyn_image.get_time_frame_definitions()).get_start_time(frame_num)+0.5*(dyn_image.get_time_frame_definitions()).get_duration(frame_num) << "\t\t" 
     //	<< << "\t\t" 
	<< patlak_x_RoI[frame_num-1] << "\t\t" 
	<< patlak_y_RoI[frame_num-1] << "\n" ; 
 out.close();   
}
// Writing images to file
//  shared_ptr<OutputFileFormat> output_format_sptr =
//    new DefaultOutputFileFormat;

  std::cerr << "Writing 'y_intersection'-image: " << y_intersection_string << "\n";

  Succeeded slope_success =
  write_basic_interfile(y_intersection_string, *y_intersection_image_sptr);
  std::cerr << "Writing 'slope'-image: " << slope_string << "\n";  
 Succeeded y_intersection_success = 
 write_basic_interfile(slope_string, *slope_image_sptr);

//output_format_sptr->write_to_file(slope_string, *slope_image_sptr);

//output_format_sptr->write_to_file(y_intersection_string, *y_intersection_image_sptr) 
if (y_intersection_success==Succeeded::yes && slope_success==Succeeded::yes)
  return EXIT_SUCCESS ;
else 
return EXIT_FAILURE ;	
}


