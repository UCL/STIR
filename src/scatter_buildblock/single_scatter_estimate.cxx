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

static const float total_cross_section_511keV = 
  total_cross_section(511.); 

float scatter_estimate_for_all_scatter_points(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,		
	  const bool use_cosphi,
	  const bool use_cache,
	  const int scatter_level)	
{	
	double scatter_ratio=0, scatter_ratio_2 = 0; 
		
	const VoxelsOnCartesianGrid<float>& image =
		dynamic_cast<const VoxelsOnCartesianGrid<float>&>(image_as_density);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	const float scatter_volume = voxel_size[1]*voxel_size[2]*voxel_size[3];
	
	for(std::size_t scatter_point_num =0;
		scatter_point_num < scatt_points_vector.size();
	    ++scatter_point_num)
	{
		if(scatter_level!=2)
			scatter_ratio +=
			scatter_estimate_for_one_scatter_point(
			image_as_activity, image_as_density, 
			scatter_point_num,
			det_num_A, det_num_B,
			lower_energy_threshold, 
			upper_energy_threshold,		
			use_cosphi, use_cache);	
		if(scatter_level!=1)
			for(std::size_t scatter_point_2_num =0;
			scatter_point_2_num < scatt_points_vector.size() ;
			++scatter_point_2_num)			
				if(scatter_point_2_num!=scatter_point_num)
					scatter_ratio_2 +=
					scatter_estimate_for_two_scatter_points(
					image_as_activity, image_as_density, 
					scatter_point_num,
					scatter_point_2_num,
					det_num_A, det_num_B,
					lower_energy_threshold, 
					upper_energy_threshold,		
					use_cosphi, use_cache);
	}	
	return scatter_volume/total_cross_section_511keV*
		(scatter_ratio+
		scatter_ratio_2
	//	*scatter_volume
		/total_cross_section_511keV);
}

END_NAMESPACE_STIR
