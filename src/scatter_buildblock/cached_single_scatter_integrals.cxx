//
// $Id$
//
/*!
\file
\ingroup scatter
\brief Implementations of functions defined in Scatter.h
Function calculates the integral along LOR in an image (attenuation or emission). 
(From scatter point to detector coordinate)

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  	  
		$Date$
		$Revision$
		
		  Copyright (C) 2004- $Date$, Hammersmith Imanet
		  See STIR/LICENSE.txt for details
*/
#include "local/stir/Scatter.h"
#include "stir/IndexRange.h" 
#include "stir/Array.h"
#include "stir/Coordinate3D.h"
#include "stir/CartesianCoordinate2D.h" 
START_NAMESPACE_STIR

float cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  			     const unsigned scatter_point_num, 
					 const unsigned det_num,
					 const image_type input_image_type)
{		 
  static IndexRange<3> range (Coordinate3D<int> (0,0,0), 
			      Coordinate3D<int> (1,
						 static_cast<int>(scatt_points_vector.size()-1),
						 total_detectors-1));
  static Array<3,float> 
    cached_integral_scattpoint_det(range);
  static bool to_initialize_array = true;
	if(to_initialize_array)		
	{
		cached_integral_scattpoint_det.fill(-1.F);
		to_initialize_array = false;
	}	
	if(input_image_type==act_image_type)
	{	
		if (cached_integral_scattpoint_det[0][scatter_point_num][det_num]<0)
		  {
		    cached_integral_scattpoint_det[0][scatter_point_num][det_num]=
		      integral_over_activity_image_between_scattpoint_det(
					      discretised_image,	  	
					      scatt_points_vector[scatter_point_num].coord,
					      detection_points_vector[det_num]);
		  }
		return 
		  cached_integral_scattpoint_det[0][scatter_point_num][det_num];
	}
	else if(input_image_type==att_image_type)
	{
#ifndef NEWSCALE		
	/* projectors work in pixel units, so convert attenuation data 
	   from cm^-1 to pixel_units^-1 */
  const float	rescale = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float> &>(discretised_image).
		get_grid_spacing()[3]/10;
#else
  const float	rescale = 
		0.1F;
#endif
		if (cached_integral_scattpoint_det[1][scatter_point_num][det_num]<0)
		  {
		    cached_integral_scattpoint_det[1][scatter_point_num][det_num]=
		      exp(-rescale*
			      integral_scattpoint_det(
						   discretised_image,	  	
						   scatt_points_vector[scatter_point_num].coord,
						   detection_points_vector[det_num]));
		  }
		return 
		  cached_integral_scattpoint_det[1][scatter_point_num][det_num];
	}		
	else 
	  {
	    error("cached_factors: wrong type of image\n");
	    return 0;// to avoid compiler warnings, but will never get here
	  }
}	
END_NAMESPACE_STIR
