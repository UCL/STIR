//
// $Id$
//
/*!
\file
\ingroup scatter
\brief Implementations of functions defined in scatter.h
Function calculates the integral along LOR in a 
image (attenuation or emission). 
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
#include "stir/CartesianCoordinate2D.h" 

START_NAMESPACE_STIR

float cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  			     const unsigned scatter_point_num, 
					 const unsigned det_num,
					 const image_type input_image_type)
{		 
	IndexRange<3> range(CartesianCoordinate3D<int> (0,0,0), CartesianCoordinate3D<int> (1,2224,71));
	static Array<3,float> cached_integral_scattpoint_det(range);
	static bool to_initialize_array = true;
	if(to_initialize_array)		
	{
		cached_integral_scattpoint_det.fill(-1.F);
		to_initialize_array = false;
	}

	if(input_image_type==act_image_type)
	{	
		if (cached_integral_scattpoint_det[0][scatter_point_num][det_num]!=-1.F)
			return cached_integral_scattpoint_det[0][scatter_point_num][det_num];
		else 
			cached_integral_scattpoint_det[0][scatter_point_num][det_num]=
			integral_scattpoint_det(
			discretised_image,	  	
			scatt_points_vector[scatter_point_num].coord,
			detection_points_vector[det_num]);			
		return 
			cached_integral_scattpoint_det[0][scatter_point_num][det_num];
	}
	
	if(input_image_type==att_image_type)
	{
		if (cached_integral_scattpoint_det[1][scatter_point_num][det_num]!=-1.F)
			return cached_integral_scattpoint_det[1][scatter_point_num][det_num];
		else 
			cached_integral_scattpoint_det[1][scatter_point_num][det_num]=
			exp(-integral_scattpoint_det(
			discretised_image,	  	
			scatt_points_vector[scatter_point_num].coord,
			detection_points_vector[det_num]));			
		return 
			cached_integral_scattpoint_det[1][scatter_point_num][det_num];
	}		
	else 
		return EXIT_FAILURE;
}	
END_NAMESPACE_STIR
