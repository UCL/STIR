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
#include "local/stir/ScatterEstimationByBin.h"
#include "stir/round.h"
#include <math.h>
using namespace std;
START_NAMESPACE_STIR

static const float total_cross_section_511keV = 
ScatterEstimationByBin::
  total_cross_section(511.F); 

float
ScatterEstimationByBin::
 scatter_estimate_for_one_scatter_point(
	  const std::size_t scatter_point_num, 
	  const unsigned det_num_A, 
	  const unsigned det_num_B)
{	
	static const float max_single_scatter_cos_angle=max_cos_angle(lower_energy_threshold,2.,energy_resolution);

	//static const float min_energy=energy_lower_limit(lower_energy_threshold,2.,energy_resolution);

	const CartesianCoordinate3D<float>& scatter_point =
	  scatt_points_vector[scatter_point_num].coord;
	const CartesianCoordinate3D<float>& detector_coord_A =
	  detection_points_vector[det_num_A];
	const CartesianCoordinate3D<float>& detector_coord_B =
	  detection_points_vector[det_num_B];
	// note: costheta is -cos_angle such that it is 1 for zero scatter angle
	const float costheta =
	  -cos_angle(detector_coord_A - scatter_point,
		     detector_coord_B - scatter_point);
	// note: costheta is identical for scatter to A or scatter to B
	// Hence, the cross_section and energy are identical for both cases as well.
	if(max_single_scatter_cos_angle>costheta)
		return 0;
	const float new_energy =
	  energy_after_scatter_511keV(costheta);

	const float detection_efficiency_scatter =
	  detection_efficiency(new_energy);
	if (detection_efficiency_scatter==0)
		return 0;

	const float emiss_to_detA =
	  cached_integral_over_activity_image_between_scattpoint_det
	  (scatter_point_num, 
	   det_num_A);		
	const float emiss_to_detB = 
	  cached_integral_over_activity_image_between_scattpoint_det
	  (scatter_point_num, 
	   det_num_B);
	if (emiss_to_detA==0 && emiss_to_detB==0)
	  return 0;	
	const float atten_to_detA = 
	  cached_exp_integral_over_attenuation_image_between_scattpoint_det
	  (scatter_point_num, 
	   det_num_A);
	const float atten_to_detB = 
	  cached_exp_integral_over_attenuation_image_between_scattpoint_det
	  (scatter_point_num, 
	   det_num_B);

	const float dif_cross_section_value =
	  dif_cross_section(costheta, 511.F); 
	
	const float rA_squared=norm_squared(scatter_point-detector_coord_A);
	const float rB_squared=norm_squared(scatter_point-detector_coord_B);

	const float scatter_point_mu=
		scatt_points_vector[scatter_point_num].mu_value;

#ifndef NDEBUG		
	const DiscretisedDensityOnCartesianGrid<3,float>& image =
	dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>&>
	  (*density_image_sptr);
	const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
	CartesianCoordinate3D<float>  origin = 
		image.get_origin();
	origin.z() -= 
		(image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;

	assert(scatter_point_mu==
		image[round((scatter_point-origin)/voxel_size)]);
#endif	          

	float scatter_ratio=0 ;

        	scatter_ratio= 
		  (emiss_to_detA*(1.F/rB_squared)*pow(atten_to_detB,total_cross_section_relative_to_511keV(new_energy)-1) 
		   +emiss_to_detB*(1.F/rA_squared)*pow(atten_to_detA,total_cross_section_relative_to_511keV(new_energy)-1)) 
		  *atten_to_detB
		  *atten_to_detA
		  *scatter_point_mu
		  *detection_efficiency_scatter;
		

	    const CartesianCoordinate3D<float> 
	      detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	    const CartesianCoordinate3D<float> 
	      detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	    const float cos_incident_angle_AS = 
	      cos_angle(scatter_point - detector_coord_A,
			detA_to_ring_center) ;
	    const float cos_incident_angle_BS = 
	      cos_angle(scatter_point - detector_coord_B,
			detB_to_ring_center) ;
	    
		return scatter_ratio*cos_incident_angle_AS*cos_incident_angle_BS*dif_cross_section_value;
	  
}

END_NAMESPACE_STIR
