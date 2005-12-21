
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

START_NAMESPACE_STIR

void apply_patlak_to_images_and_arterial_sampling(DiscretisedDensity<3,float>& y_intersection_image, 			     
			    DiscretisedDensity<3,float>& slope_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected)
{
  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();

  // Apply calibration_factor and Decay Correct Frame Image
  if(!is_calibrated)
    error("The input image seems not to be calibrated");
  dyn_image.calibrate_frames();    

  if(!is_decay_corrected)
    {
      dyn_image.set_isotope_halflife(6586.2F); //FDG halflife in seconds
      dyn_image.decay_correct_frames(); 
      plasma_data.set_isotope_halflife(6586.2F);
      plasma_data.decay_correct_PlasmaData(); 
    }

  // Initialise the vector where the plasma values for each frame to 0. 
  std::vector<float>  plasma_vector(num_frames,0) ; 
  std::vector<float>  blood_vector(num_frames,0) ; 
  std::vector<float>  plasma_sum_vector(num_frames,0) ; //I use sum instead of mean, since PET frames estimate the sum.
  float plasma_frame_sum=0; //used to store the previous frame values
  PlasmaData::const_iterator cur_iter;
  unsigned int frame_num;
  // float previous_time=0; // Trick to store border times
  // float next_time=0;  // Trick to store border times

  //Short plasma data in frames
  // Estimate the plasma_frame_vector and the plasma_frame_sum_vector using th linear_integral() implementation
  for (  frame_num = 1; 
	 frame_num<=num_frames ; ++frame_num )
    {     
      std::vector<float> time_frame_vector ; 
      std::vector<float> plasma_frame_vector ;
      std::vector<float> blood_frame_vector ;
      //Create border value using linear interpolation: y=y1+(y2-y1)*(t-t1)/(t2-t1)
      const float frame_start_time=(dyn_image.get_time_frame_definitions()).get_start_time(frame_num);//t1
      const float frame_end_time=(dyn_image.get_time_frame_definitions()).get_end_time(frame_num);//t2
    
      //  if(frame_num!=1)
      //	{
      //  time_frame_vector.push_back(frame_start_time);
      //  plasma_frame_vector.push_back((next_plasma_cnt-previous_plasma_cnt)*(next_time-frame_start_time)/(next_time-previous_time));
      // blood_frame_vector.push_back((next_blood_cnt-previous_blood_cnt)*(next_time-frame_start_time)/(next_time-previous_time));
      //	}

      for( cur_iter=plasma_data.begin() ; cur_iter!=plasma_data.end() && cur_iter->get_time_in_s()<frame_end_time ; ++cur_iter)
    	{
	  const float cur_time=(*cur_iter).get_time_in_s() ;

	  if (cur_time<frame_start_time)
	    continue;
	  const float cur_plasma_cnt=(*cur_iter).get_plasma_counts_in_kBq();
	  const float cur_blood_cnt=(*cur_iter).get_blood_counts_in_kBq()  ;
	  if (cur_time<frame_end_time)
	    {
	      plasma_frame_vector.push_back(cur_plasma_cnt);
	      blood_frame_vector.push_back(cur_blood_cnt);
	      time_frame_vector.push_back(cur_time);	    
	    }
	  /*
	    if (frame_num!=num_frames)
	    if (cur_time>(dyn_image.get_time_frame_definitions()).get_end_time(frame_num+1))
	    break;*/
	  else
	    //{
	    //previous_time=cur_time;
	    //previous_plasma_cnt=cur_plasma_cnt;
	    //previous_blood_cnt=cur_blood_cnt;
	    //if(cur_iter!=plasma_data.end()-1 && frame_num!=num_frames)
	    //{
	    //      next_time=(cur_iter+1)->get_time_in_s();
	    //      next_plasma_cnt=(cur_iter+1)->get_plasma_counts__in_kBq();
	    //      next_blood_cnt=(cur_iter+1)->get_blood_counts_in_kBq();
	    //      time_frame_vector.push_back(frame_start_time);
	    // plasma_frame_vector.push_back((next_plasma_cnt-previous_plasma_cnt)*(next_time-frame_start_time)/(next_time-previous_time));
	    // blood_frame_vector.push_back((next_blood_cnt-previous_blood_cnt)*(next_time-frame_start_time)/(next_time-previous_time));
	    //	}
	    break;
	  //  }
	  //Create border value using linear interpolation: y=y1+(y2-y1)*(t-t1)/(t2-t1)
	  
	}
      //      plasma_frame_vector[frame_num-1]=push_back(plasma_frame_vector[frame_num-1]);
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
  std::vector<float> tissue_vector_RoI(num_frames);
  std::vector<float> plasma_vector_RoI(num_frames);
  int total_voxels_RoI=0;
  // Do linear_regression for each voxel 
  // for k j i 
  {
    float slope=0.F;
    float y_intersection=0.F;
    float variance_of_slope=0.F;
    float variance_of_y_intersection=0.F;
    float covariance_of_y_intersection_with_slope=0.F;
    float chi_square = 0.F;  
     
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
		for ( frame_num = starting_frame; // shifting time because in early points, plasma_vector can be 0.// Or is it important?
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
			// std::cerr << "location selected [" << k << "] ["  << j << "] ["  << i << "]\n " ;
			plasma_vector_RoI[frame_num-1]+=plasma_vector[frame_num-1];
			patlak_y_RoI[frame_num-1]+=patlak_y[frame_num-1];
			patlak_x_RoI[frame_num-1]+=patlak_x[frame_num-1];
			tissue_vector_RoI[frame_num-1]+=dyn_image[frame_num][k][j][i];
			++total_voxels_RoI;
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
  }

/*	
  // Writing TAC to file.
  std::cerr << "Testing TAC. Look at the .tac files!\n";	  
  string tac_string=input_string+".tac";	 
  std::ofstream out(tac_string.c_str()); //output file //
  if(!out)
    std::cout << "Cannot open text file.\n" ; 
  out << "Frame" << "\tTimePoint\t" << "\tPlasma\t" << "\tTissue\t" << "\tRoI-X\t" << "\tRoI-Y\n" ;
    
  for (  frame_num = 1 ; frame_num<=num_frames ; ++frame_num )
    out  << frame_num << "\t"
	 << (dyn_image.get_time_frame_definitions()).get_start_time(frame_num)+0.5*(dyn_image.get_time_frame_definitions()).get_duration(frame_num) << "\t\t" 
	 << plasma_vector_RoI[frame_num-1]/total_voxels_RoI << "\t\t"
	 << tissue_vector_RoI[frame_num-1]/total_voxels_RoI << "\t\t"
	 << patlak_x_RoI[frame_num-1]/total_voxels_RoI << "\t\t" 
	 << patlak_y_RoI[frame_num-1]/total_voxels_RoI << "\n" ; 
  out.close(); 
 }*/
}

void apply_patlak_to_images_plasma_based(DiscretisedDensity<3,float>& y_intersection_image, 			     
			    DiscretisedDensity<3,float>& slope_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    BloodFrameData & blood_frame_data,
			    const int starting_frame, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected)
{
  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();

  // Apply calibration_factor and Decay Correct Frame Image
  if(!is_calibrated)
    error("The input image seems not to be calibrated");
  dyn_image.calibrate_frames();    

  if(!is_decay_corrected)
    {
      dyn_image.set_isotope_halflife(6586.2F); //FDG halflife in seconds
      dyn_image.decay_correct_frames(); 
      blood_frame_data.set_isotope_halflife(6586.2F);
      blood_frame_data.decay_correct_BloodFrameData(); 
    }

  // Initialise the vector where the blood values for each frame to 0. 
  std::vector<float>  blood_vector(num_frames,0); 
  std::vector<float>  blood_sum_vector(num_frames,0) ;
  float blood_frame_sum=0; //used to store the previous frame values
  BloodFrameData::const_iterator cur_iter=blood_frame_data.begin();
  unsigned int frame_num;

  for (  frame_num = 1; 
	 frame_num<=num_frames ; ++frame_num )
    {         
      blood_vector[frame_num-1]=cur_iter->get_blood_counts_in_kBq();
      // The reconstructed CTI images and the STIR (when using reconstruction script) are always devided by the time frame duration.;
      blood_frame_sum+=(cur_iter->get_frame_end_time_in_s()-cur_iter->get_frame_start_time_in_s())*cur_iter->get_blood_counts_in_kBq();
      blood_sum_vector[frame_num-1]=blood_frame_sum ;
      std::cerr << "Frame_Value: " << blood_vector[frame_num-1] << "   \n";
      std::cerr << "Sum: " << blood_sum_vector[frame_num-1] << "   \n";
      ++cur_iter;
    }
  std::vector<float> patlak_y_RoI(num_frames);
  std::vector<float> patlak_x_RoI(num_frames);
  std::vector<float> tissue_vector_RoI(num_frames);
  std::vector<float> blood_vector_RoI(num_frames);
  int total_voxels_RoI=0;
  // Do linear_regression for each voxel 
  // for k j i 
  {
    float slope=0.F;
    float y_intersection=0.F;
    float variance_of_slope=0.F;
    float variance_of_y_intersection=0.F;
    float covariance_of_y_intersection_with_slope=0.F;
    float chi_square = 0.F;  
     
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
		for ( frame_num = starting_frame; // shifting time because in early points, plasma_vector can be 0.// Or is it important?
		      frame_num<=num_frames ; ++frame_num )
		  {
		    patlak_y[frame_num-1]=(dyn_image[frame_num][k][j][i]/blood_vector[frame_num-1]);
		    patlak_x[frame_num-1]=blood_sum_vector[frame_num-1]/blood_vector[frame_num-1];
		    weights[frame_num-1]=1;			
			
		    if(abs(max_k_index+min_k_index-2*k)<=6 && 
		       abs(max_j_index+min_j_index-2*j)<=6 && 
		       abs(max_i_index+min_i_index-2*i)<=6)
		      {
			// std::cerr << "location selected [" << k << "] ["  << j << "] ["  << i << "]\n " ;
			blood_vector_RoI[frame_num-1]+=blood_vector[frame_num-1];
			patlak_y_RoI[frame_num-1]+=patlak_y[frame_num-1];
			patlak_x_RoI[frame_num-1]+=patlak_x[frame_num-1];
			tissue_vector_RoI[frame_num-1]+=dyn_image[frame_num][k][j][i];
			++total_voxels_RoI;
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
  }
/*	
  // Writing TAC to file.
  std::cerr << "Testing TAC. Look at the .tac files!\n";	  
  string tac_string=input_string+".tac";	 
  std::ofstream out(tac_string.c_str()); //output file //
  if(!out)
    std::cout << "Cannot open text file.\n" ; 
  out << "Frame" << "\tTimePoint\t" << "\tPlasma\t" << "\tTissue\t" << "\tRoI-X\t" << "\tRoI-Y\n" ;
    
  for (  frame_num = 1 ; frame_num<=num_frames ; ++frame_num )
    out  << frame_num << "\t"
	 << (dyn_image.get_time_frame_definitions()).get_start_time(frame_num)+0.5*(dyn_image.get_time_frame_definitions()).get_duration(frame_num) << "\t\t" 
	 << plasma_vector_RoI[frame_num-1]/total_voxels_RoI << "\t\t"
	 << tissue_vector_RoI[frame_num-1]/total_voxels_RoI << "\t\t"
	 << patlak_x_RoI[frame_num-1]/total_voxels_RoI << "\t\t" 
	 << patlak_y_RoI[frame_num-1]/total_voxels_RoI << "\n" ; 
  out.close(); 
 }*/
}
  END_NAMESPACE_STIR
