//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in scatter.h

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans

  $Date$
  $Revision$

    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "local/stir/Scatter.h"

using namespace std;

START_NAMESPACE_STIR
float scatter_estimate_for_all_scatter_points(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,		
	  const bool use_cosphi,
	  const bool use_cache)	
{	
	float scatter_ratio = 0; 
	
	for(std::size_t scatter_point_num =0;
		scatter_point_num < scatt_points_vector.size();
	    ++scatter_point_num)
	  {
		  scatter_ratio +=
			  scatter_estimate_for_one_scatter_point(
			  image_as_activity, image_as_density, 
			  scatter_point_num,
			  det_num_A, det_num_B,
			  lower_energy_threshold, 
			  upper_energy_threshold,		
			  use_cosphi, use_cache);
	  }
    
	return scatter_ratio;
}


END_NAMESPACE_STIR
