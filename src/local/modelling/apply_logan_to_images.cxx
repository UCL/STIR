
//
// $id: apply_logan_on_images.cxx,v 1.1 2005/12/02 16:22:23 ctsoumpas Exp $
//
/*!
  \file
  \ingroup utilities
  \brief Apply the Logan linear fit using Dynamic Images
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

#include "stir/TimeFrameDefinitions.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/modelling/PlasmaSample.h"
#include "local/stir/modelling/logan.h"
#include "local/stir/modelling/LoganPlot.h"
#include "local/stir/modelling/ModelMatrix.h"
#include "local/stir/numerics/linear_integral.h"
#include "stir/linear_regression.h"
#include "stir/Array.h"
#include "stir/VectorWithOffset.h"
#include "stir/utilities.h"

START_NAMESPACE_STIR

void apply_OLS_logan_to_images(ParametricVoxelsOnCartesianGrid & par_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const unsigned int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected)
{
  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  // Apply calibration_factor and Decay Correct Frame Image
  if(!is_calibrated)
    error("The input image seems not to be calibrated");
  dyn_image.calibrate_frames();    
#if 1
  if(!is_decay_corrected)
    {
      dyn_image.set_isotope_halflife(6586.2F); //FDG halflife in seconds
      dyn_image.decay_correct_frames(); 
      plasma_data.set_isotope_halflife(6586.2F);
      plasma_data.decay_correct_PlasmaData(); 
    }
#endif
  PlasmaData::const_iterator cur_iter;
  unsigned int frame_num;
  PlasmaData sample_data_in_frames(plasma_data.get_sample_data_in_frames(dyn_image.get_time_frame_definitions()));
  LoganPlot brain_logan_plot;
  Array<2,float> brain_logan_model_array=
    (brain_logan_plot.get_model_matrix(
					sample_data_in_frames,dyn_image.get_time_frame_definitions(),starting_frame)).get_model_array();
  VectorWithOffset<float> logan_x(starting_frame-1,num_frames-1);
  VectorWithOffset<float> logan_y(starting_frame-1,num_frames-1); 
  VectorWithOffset<float> weights(starting_frame-1,num_frames-1);
  for(unsigned int frame_num = starting_frame; // shifting time because in early points, plasma_vector can be 0.// Or is it important?
      frame_num<=num_frames ; ++frame_num )
      weights[frame_num-1]=1;			 

  {  // Do linear_regression for each voxel // for k j i 
    float slope=0.F;
    float y_intersection=0.F;
    float variance_of_slope=0.F;
    float variance_of_y_intersection=0.F;
    float covariance_of_y_intersection_with_slope=0.F;
    float chi_square = 0.F;  
     
    const int min_k_index = dyn_image[1].get_min_index(); 
    const int max_k_index = dyn_image[1].get_max_index();
    for ( int k = min_k_index; k<= max_k_index; ++k)
      {
	const int min_j_index = dyn_image[1][k].get_min_index(); 
	const int max_j_index = dyn_image[1][k].get_max_index();
	for ( int j = min_j_index; j<= max_j_index; ++j)
	  {
	    const int min_i_index = dyn_image[1][k][j].get_min_index(); 
	    const int max_i_index = dyn_image[1][k][j].get_max_index();
	    for ( int i = min_i_index; i<= max_i_index; ++i)
	      {	
		float tissue_integral=0;
		for ( frame_num = starting_frame; // shifting time because in early points, plasma_vector can be 0.// Or is it important?
		      frame_num<=num_frames ; ++frame_num )
		  {
		    const float frame_duration=(dyn_image.get_time_frame_definitions()).get_duration(frame_num);

		    const float tissue_activity=dyn_image[frame_num][k][j][i]-bv*brain_logan_model_array[2][frame_num];
		    tissue_integral+=tissue_activity*frame_duration; 

		    logan_y[frame_num-1]=tissue_integral/tissue_activity;
		    logan_x[frame_num-1]=brain_logan_model_array[1][frame_num]/tissue_activity;
		  }
		linear_regression(y_intersection, slope,
				  chi_square,
				  variance_of_y_intersection,
				  variance_of_slope,
				  covariance_of_y_intersection_with_slope,
				  logan_y,
				  logan_x,		      
				  weights);	
		par_image[k][j][i][2]=-1/y_intersection;
		par_image[k][j][i][1]=slope;
	      }
	  }
      }    
  }
}

void apply_NLLS_logan_to_images(ParametricVoxelsOnCartesianGrid & par_image, 
			    DynamicDiscretisedDensity & dyn_image,
			    PlasmaData & plasma_data,
			    const float time_shift, 
			    const unsigned int starting_frame, 
			    const float bv, 
			    const bool is_calibrated , 
			    const bool is_decay_corrected)
{}
END_NAMESPACE_STIR

