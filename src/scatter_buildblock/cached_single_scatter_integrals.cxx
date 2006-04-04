//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in stir::ScatterEstimationByBin

  Function calculates the integral along LOR in an image (attenuation or emission). 
  (From scatter point to detector coordinate)

  \author Charalampos Tsoumpas
  \author Nikolaos Dikaios
  \author Kris Thielemans
  	  
  $Date$
  $Revision$
		
  Copyright (C) 2004- $Date$, Hammersmith Imanet
  See STIR/LICENSE.txt for details
*/
#include "local/stir/ScatterEstimationByBin.h"
#include "stir/IndexRange.h" 
#include "stir/Coordinate2D.h"
START_NAMESPACE_STIR

const float cache_init_value = -1.234F; // an arbitrary value that should never occur

void
ScatterEstimationByBin::
initialise_cache_for_scattpoint_det()
{
  IndexRange<2> range (Coordinate2D<int> (0,0), 
		       Coordinate2D<int> (static_cast<int>(this->scatt_points_vector.size()-1),
					  this->total_detectors-1));

  this->cached_activity_integral_scattpoint_det.resize(range);
  this->cached_attenuation_integral_scattpoint_det.resize(range);

  this->cached_activity_integral_scattpoint_det.fill(cache_init_value);
  this->cached_attenuation_integral_scattpoint_det.fill(cache_init_value);
}

float 
ScatterEstimationByBin::
cached_integral_over_activity_image_between_scattpoint_det(const unsigned scatter_point_num, 
							   const unsigned det_num)
{		 
  float * location_in_cache = 
    this->use_cache
    ? &cached_activity_integral_scattpoint_det[scatter_point_num][det_num]
    : 0;

  if (this->use_cache && *location_in_cache!=cache_init_value)
    {
      return *location_in_cache;
    }
  else
    {
      const float result =
	integral_over_activity_image_between_scattpoint_det
	(scatt_points_vector[scatter_point_num].coord,
	 detection_points_vector[det_num]
	 );
      if (this->use_cache)
	*location_in_cache=result;
      return result;
    }
}

float 
ScatterEstimationByBin::
cached_exp_integral_over_attenuation_image_between_scattpoint_det(const unsigned scatter_point_num, 
								  const unsigned det_num)
{
  float * location_in_cache = 
    this->use_cache
    ? &cached_attenuation_integral_scattpoint_det[scatter_point_num][det_num]
    : 0;

  if (this->use_cache && *location_in_cache!=cache_init_value)
    {
      return *location_in_cache;
    }
  else
    {
      const float result =
	exp_integral_over_attenuation_image_between_scattpoint_det
	(scatt_points_vector[scatter_point_num].coord,
	 detection_points_vector[det_num]
	 );
      if (this->use_cache)
	*location_in_cache=result;
      return result;
    }
}
	
END_NAMESPACE_STIR
