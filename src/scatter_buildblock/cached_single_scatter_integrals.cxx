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

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	
	  
		$Date$
		$Revision$
		
		  Copyright (C) 2004- $Date$, Hammersmith Imanet
		  See STIR/LICENSE.txt for details
*/


#include "local/stir/Scatter.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <map>
#include <utility>
#include <functional>


START_NAMESPACE_STIR

template <class coordT>
bool 
operator<(const Coordinate3D<coordT>& p1, 
          const Coordinate3D<coordT>& p2) 
{
	return 
		p1[1]<p2[1] ||
		(p1[1]==p2[1]&& 
		(p1[2]<p2[2] || 
		(p1[2]==p2[2] && 
		(p1[3]<p2[3]))));
}

template <class coordT>
bool 
operator==(const Coordinate3D<coordT>& p1, 
		   const Coordinate3D<coordT>& p2) 
{
	return 
		p1[1]==p2[1] && p1[2]==p2[2] && p1[3]==p2[3];
}

typedef std::pair<CartesianCoordinate3D<float>,
CartesianCoordinate3D<float> > key_type;

float cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  			     const CartesianCoordinate3D<float>& scatter_point, 
				     const CartesianCoordinate3D<float>& detector_coord,
					 const int input_image_type)
{				

	typedef std::map<key_type,float> map_type;
		const VoxelsOnCartesianGrid<float>& image =
			dynamic_cast<const VoxelsOnCartesianGrid<float>& >(discretised_image);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	if(input_image_type==1)
	{
	    static map_type cached_integral_act_scattpoint_det;		
		const key_type key= std::make_pair(scatter_point, detector_coord);
		{
			// test if key is present in map, if so, return its value
			map_type::const_iterator  iter =
				cached_integral_act_scattpoint_det.find(key);
			if (iter != cached_integral_act_scattpoint_det.end())
				return iter->second;
		}				
		cached_integral_act_scattpoint_det[key] = integral_scattpoint_det(
			discretised_image,	  	
			scatter_point, 
			detector_coord);			
		return cached_integral_act_scattpoint_det[key];
	}	

	if(input_image_type==0)
	{	
		static map_type cached_integral_att_scattpoint_det;	
		const key_type key= std::make_pair(scatter_point, detector_coord);
		{
			// test if key is present in map, if so, return its value
			map_type::const_iterator  iter =
				cached_integral_att_scattpoint_det.find(key);
			if (iter != cached_integral_att_scattpoint_det.end())
				return iter->second;
		}						
		cached_integral_att_scattpoint_det[key] =integral_scattpoint_det(
			discretised_image,	  	
			scatter_point, 
			detector_coord);			
		return cached_integral_att_scattpoint_det[key];
	}		
	else 
		return EXIT_FAILURE;
}						   
END_NAMESPACE_STIR

