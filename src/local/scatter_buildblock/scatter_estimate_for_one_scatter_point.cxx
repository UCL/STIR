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
#include <math.h>
using namespace std;
START_NAMESPACE_STIR

static const float total_cross_section_511keV = 
  total_cross_section(511.); 

float scatter_estimate_for_one_scatter_point(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const std::size_t scatter_point_num, 
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,
	  const float resolution,		
	  const bool use_cache, const bool use_sintheta)
{	
	static const float max_single_scatter_cos_angle=max_cos_angle(lower_energy_threshold,2.,resolution);

	static const float min_energy=energy_lower_limit(lower_energy_threshold,2.,resolution);

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
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   new_energy,511.F,resolution);
	if (detection_efficiency_scatter==0)
		return 0;

	float emiss_to_detA, 
		emiss_to_detB,
		atten_to_detA,
		atten_to_detB;

	if (use_cache)
	{			
		emiss_to_detA = cached_factors(
			image_as_activity,
			scatter_point_num, 
			det_num_A
			, act_image_type);		
		emiss_to_detB = cached_factors(
			image_as_activity,
			scatter_point_num, 
			det_num_B
			,act_image_type);
		if (emiss_to_detA==0 && emiss_to_detB==0)
		return 0;	
		atten_to_detA = cached_factors(
			image_as_density,
			scatter_point_num, 
			det_num_A
			, att_image_type);
		atten_to_detB = cached_factors(
			image_as_density,
			scatter_point_num, 
			det_num_B
			, att_image_type);
	}
	else
	{
		emiss_to_detA = integral_scattpoint_det( 
			image_as_activity,
			scatter_point, 
			detector_coord_A);
		emiss_to_detB = integral_scattpoint_det(
			image_as_activity,
			scatter_point, 
			detector_coord_B);
		if (emiss_to_detA==0 && emiss_to_detB==0)
			return 0;	

	
#ifndef NEWSCALE		
	/* projectors work in pixel units, so convert attenuation data 
	   from cm^-1 to pixel_units^-1 */
		const float	rescale = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float> &>(image_as_density).
		get_grid_spacing()[3]/10;
#else
  const float	rescale = 
		0.1F;
#endif
		atten_to_detA = exp(-rescale*integral_scattpoint_det(
			image_as_density,
			scatter_point, 
			detector_coord_A));
		atten_to_detB = exp(-rescale*integral_scattpoint_det(
			image_as_density,
			scatter_point, 
			detector_coord_B));
	}	
	const float dif_cross_section =
		dif_cross_section_511keV(costheta); 

	const float dif_cross_section_sin =
		dif_cross_section_sin_511keV(costheta); 
	
	const float rA_squared=norm_squared(scatter_point-detector_coord_A);
	const float rB_squared=norm_squared(scatter_point-detector_coord_B);
	
	const float scatter_point_mu=
		scatt_points_vector[scatter_point_num].mu_value;

#ifndef NDEBUG		
	const VoxelsOnCartesianGrid<float>& image =
		static_cast<const VoxelsOnCartesianGrid<float>&>(image_as_density);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	CartesianCoordinate3D<float>  origin = 
		image.get_origin();
	origin.z() -= 
		(image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;

	assert(scatter_point_mu==
		image[round((scatter_point-origin)/voxel_size)]);
#endif	          

	float scatter_ratio=0 ;

	if (use_sintheta)
        	scatter_ratio= 
		(emiss_to_detA*pow(atten_to_detB,total_cross_section_relative_to_511keV(new_energy)-1) 
		+emiss_to_detB*pow(atten_to_detA,total_cross_section_relative_to_511keV(new_energy)-1))
		/(rA_squared*rB_squared) 
		*dif_cross_section_sin
		*atten_to_detB
		*atten_to_detA
		*scatter_point_mu
		*detection_efficiency_scatter
		;
                 
	if (!use_sintheta)
	scatter_ratio =
		(emiss_to_detA*pow(atten_to_detB,total_cross_section_relative_to_511keV(new_energy)-1) 
		+emiss_to_detB*pow(atten_to_detA,total_cross_section_relative_to_511keV(new_energy)-1))
		/(rA_squared*rB_squared) 
		*dif_cross_section
		*atten_to_detB
		*atten_to_detA
		*scatter_point_mu
		*detection_efficiency_scatter
		;	


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
	    
	    return
	      scatter_ratio
	      *cos_incident_angle_AS
	      *cos_incident_angle_BS;
;
	
}

END_NAMESPACE_STIR
