//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h

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
	  const bool use_cache,
	  const int scatter_level)	
{	
	double single_scatter_ratio = 0, 
		double_scatter_ratio = 0,
		triple_scatter_ratio = 0; 

	// TODO: slightly dangerous to use a static here
	// it would give wrong results when the energy_thresholds are changed...	
	static const float detection_efficiency_no_scatter =
	  detection_efficiency_BGO(lower_energy_threshold,
				   upper_energy_threshold,
				   511);
	const CartesianCoordinate3D<float>& detector_coord_A =
		detection_points_vector[det_num_A];
    const CartesianCoordinate3D<float>& detector_coord_B =
		detection_points_vector[det_num_B];
	const CartesianCoordinate3D<float> 
	      detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	    const CartesianCoordinate3D<float> 
	      detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	const float rAB_squared=norm_squared(detector_coord_A-detector_coord_B);
	const float cos_incident_angle_A = 
	      cos_angle(detector_coord_B - detector_coord_A,
			detA_to_ring_center) ;
	    const float cos_incident_angle_B = 
	      cos_angle(detector_coord_A - detector_coord_B,
			detB_to_ring_center) ;
		
	const VoxelsOnCartesianGrid<float>& image =
		dynamic_cast<const VoxelsOnCartesianGrid<float>&>(image_as_density);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	const float scatter_volume = voxel_size[1]*voxel_size[2]*voxel_size[3];
	
	for(std::size_t scatter_point_num =0;
		scatter_point_num < scatt_points_vector.size();
	    ++scatter_point_num)
	{	
		if(scatter_level==1||scatter_level==12||scatter_level==10||scatter_level==120)
			single_scatter_ratio +=
			scatter_estimate_for_one_scatter_point(
			image_as_activity, image_as_density, 
			scatter_point_num,
			det_num_A, det_num_B,
			lower_energy_threshold, 
			upper_energy_threshold,		
			use_cache);	

		if(scatter_level==2||scatter_level==12||scatter_level==120)
			for(std::size_t scatter_point_2_num =0;
			scatter_point_2_num < scatt_points_vector.size() ;
			++scatter_point_2_num)			
				if(scatter_point_2_num!=scatter_point_num)
					double_scatter_ratio +=
					scatter_estimate_for_two_scatter_points(
					image_as_activity, image_as_density, 
					scatter_point_num,
					scatter_point_2_num,
					det_num_A, det_num_B,
					lower_energy_threshold, 
					upper_energy_threshold,		
					use_cache);
	}	
	return 0.75*rAB_squared*scatter_volume/total_cross_section_511keV/detection_efficiency_no_scatter*
		(single_scatter_ratio+double_scatter_ratio*scatter_volume/total_cross_section_511keV)
		/(cos_incident_angle_A*cos_incident_angle_A*cos_incident_angle_B*cos_incident_angle_B);
}

END_NAMESPACE_STIR
