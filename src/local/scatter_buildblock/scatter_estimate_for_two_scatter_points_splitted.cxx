//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h

  \author Charalampos Tsoumpas
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

double scatter_estimate_for_two_scatter_points_splitted(	  
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const std::size_t scatter_point_1_num, 
	  const std::size_t scatter_point_2_num, 
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold, const float resolution,
	  const bool use_cache,
          const int split)
{	
	static const float max_single_scatter_cos_angle=max_cos_angle(lower_energy_threshold,2.,resolution);
	static const float min_energy=energy_lower_limit(lower_energy_threshold,2.,resolution);

	const CartesianCoordinate3D<float>& scatter_point_1 =
		scatt_points_vector[scatter_point_1_num].coord;
	const CartesianCoordinate3D<float>& scatter_point_2 =
		scatt_points_vector[scatter_point_2_num].coord;	
	const CartesianCoordinate3D<float>& detector_coord_A =
		detection_points_vector[det_num_A];
    const CartesianCoordinate3D<float>& detector_coord_B =
		detection_points_vector[det_num_B];
	// note: costheta is -cos_angle such that it is 1 for zero scatter angle
	const float costheta_A_sc1_sc2 =
		-cos_angle(detector_coord_A - scatter_point_1,//-
		            scatter_point_2 - scatter_point_1);
	if (max_single_scatter_cos_angle>costheta_A_sc1_sc2)
		return 0;
	const float costheta_sc1_sc2_B = 
		-cos_angle( scatter_point_1 - scatter_point_2,//-
		            detector_coord_B - scatter_point_2);
	if (max_single_scatter_cos_angle>costheta_sc1_sc2_B)
		return 0;
	// note: costheta is identical for scatter to A or scatter to B
	// Hence, the cross_section and energy are identical for both cases as well.
	const float new_energy_scatter_at_1 =
	  energy_after_scatter_511keV(costheta_A_sc1_sc2);
	const float new_energy_scatter_at_2 =
	  energy_after_scatter_511keV(costheta_sc1_sc2_B);
	const float new_energy_scatter_at_1_then_2 =
	  energy_after_scatter(costheta_sc1_sc2_B,new_energy_scatter_at_1);
	const float new_energy_scatter_at_2_then_1 =
	  energy_after_scatter(costheta_A_sc1_sc2,new_energy_scatter_at_2);        
	// TODO: slightly dangerous to use a static here
	// it would give wrong results when the energy_thresholds are changed...	
	static const float detection_efficiency_no_scatter =
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   511.F,511.F,resolution);
	const float detection_efficiency_scatter_at_1 =
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   new_energy_scatter_at_1, 511.F, resolution);	
	const float detection_efficiency_scatter_at_2 =
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   new_energy_scatter_at_2, 511.F, resolution);
	const float detection_efficiency_scatter_at_1_then_2 =
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   new_energy_scatter_at_1_then_2, 511.F, resolution);	
	const float detection_efficiency_scatter_at_2_then_1 =
	  detection_efficiency(lower_energy_threshold,
				   upper_energy_threshold,
				   new_energy_scatter_at_2_then_1, 511.F, resolution);
	float emiss_to_detA, 
		emiss_to_detB,
		atten_to_detA,
		atten_to_detB,
		emiss_sc1_to_sc2,
		atten_sc1_to_sc2;
// Always use cache for the LoRs in DSS
	emiss_to_detA = cached_factors(
		image_as_activity,
		scatter_point_1_num, 
		det_num_A
		, act_image_type);			
	emiss_to_detB = cached_factors(
		image_as_activity,
		scatter_point_2_num, 
		det_num_B
		,act_image_type);
	if (use_cache)
	{
		emiss_sc1_to_sc2 = cached_factors_2(
			image_as_activity,
			scatter_point_1_num, 
			scatter_point_2_num
			, act_image_type);
		atten_sc1_to_sc2 = cached_factors_2(
			image_as_density,
			scatter_point_1_num, 
			scatter_point_2_num
			, att_image_type);
	}	
	if (!use_cache) 
	  emiss_sc1_to_sc2 = integral_scattpoint_det( 
			image_as_activity,
			scatter_point_1, 
			scatter_point_2);		
	if ((emiss_to_detA==0 || detection_efficiency_scatter_at_1_then_2 == 0)
			&& (emiss_to_detB==0 || detection_efficiency_scatter_at_2_then_1 == 0) && 
			    emiss_sc1_to_sc2==0)
//In real cases it will give small values not zero. Use of emission threshold?
			return 0;	
	if (!use_cache) 
	{
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
		atten_sc1_to_sc2 = exp(-rescale*integral_scattpoint_det(
			image_as_density,
			scatter_point_1, 
			scatter_point_2));
	}
	// Always use cache for the LoRs in DSS
		atten_to_detA = cached_factors(
			image_as_density,
			scatter_point_1_num, 
			det_num_A
			, att_image_type);
		atten_to_detB = cached_factors(
			image_as_density,
			scatter_point_2_num, 
			det_num_B
			, att_image_type);
	const double dif_cross_section_A_sc1_sc2 =
		dif_cross_section_511keV(costheta_A_sc1_sc2); 
	const double dif_cross_section_sc1_sc2_B =
		dif_cross_section_511keV(costheta_sc1_sc2_B); 
	const double dif_cross_section_A_sc1_sc2_B =
		dif_cross_section(costheta_sc1_sc2_B, new_energy_scatter_at_1); 
	const double dif_cross_section_B_sc2_sc1_A =
		dif_cross_section(costheta_A_sc1_sc2, new_energy_scatter_at_2); 	
	const float rA_squared=norm_squared(scatter_point_1-detector_coord_A);
	const float rB_squared=norm_squared(scatter_point_2-detector_coord_B);	
	const float scatter_point_1_mu=
		scatt_points_vector[scatter_point_1_num].mu_value;
	const float scatter_point_2_mu=
		scatt_points_vector[scatter_point_2_num].mu_value;
	const float total_cross_section_scatter_at_1 =
		total_cross_section_relative_to_511keV(new_energy_scatter_at_1);
	const float total_cross_section_scatter_at_2 =
		total_cross_section_relative_to_511keV(new_energy_scatter_at_2);
 
	  if(split == 0){ // 2-ScatterBlue and 0-ScatterPink + 0-ScatterBlue and 2-ScatterPink + 1-ScatterBlue and 1-ScatterPink
	     const double scatter_ratio =
		(emiss_to_detA*
		 dif_cross_section_A_sc1_sc2*
		 dif_cross_section_A_sc1_sc2_B*
//		 total_cross_section_scatter_at_2/total_cross_section_scatter_at_1*
		 detection_efficiency_scatter_at_1_then_2*
		 pow(atten_sc1_to_sc2,total_cross_section_scatter_at_1-1)*
		 pow(atten_to_detB,total_cross_section_relative_to_511keV(new_energy_scatter_at_1_then_2)-1)				 
		 +
		 emiss_to_detB*
		 dif_cross_section_sc1_sc2_B*
		 dif_cross_section_B_sc2_sc1_A*
//		 total_cross_section_scatter_at_1/total_cross_section_scatter_at_2*
		 detection_efficiency_scatter_at_2_then_1*
		 pow(atten_sc1_to_sc2,total_cross_section_scatter_at_2-1)*
		 pow(atten_to_detA,total_cross_section_relative_to_511keV(new_energy_scatter_at_2_then_1)-1)
		 +
		 emiss_sc1_to_sc2*
		 dif_cross_section_A_sc1_sc2*
		 dif_cross_section_sc1_sc2_B
		 /detection_efficiency_no_scatter
		 *detection_efficiency_scatter_at_1*detection_efficiency_scatter_at_2*
		 pow(atten_to_detA,total_cross_section_scatter_at_1-1)*
		 pow(atten_to_detB,total_cross_section_scatter_at_2-1)
		 )
		 /(rA_squared*rB_squared)
		 *atten_to_detB
		 *atten_to_detA
		 *atten_sc1_to_sc2
		 *scatter_point_1_mu
		 *scatter_point_2_mu;

	    const CartesianCoordinate3D<float> 
	      detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	    const CartesianCoordinate3D<float> 
	      detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	    const float cos_incident_angle_AS1 = 
	      cos_angle(scatter_point_1 - detector_coord_A,
			detA_to_ring_center) ;
	    const float cos_incident_angle_BS2 = 
	      cos_angle(scatter_point_2 - detector_coord_B,
			detB_to_ring_center) ;

	    return
	      scatter_ratio*cos_incident_angle_AS1*
	      //cos_incident_angle_AS1
	      //	      *cos_incident_angle_BS2*
	      cos_incident_angle_BS2 ;
	 }

 
	  if(split == 1){  // 2-ScatterBlue and 0-ScatterPink + 0-ScatterBlue and 2-ScatterPink
	     const double scatter_ratio =
		(emiss_to_detA*
		 dif_cross_section_A_sc1_sc2*
		 dif_cross_section_A_sc1_sc2_B*
//		 total_cross_section_scatter_at_2/total_cross_section_scatter_at_1*
		 detection_efficiency_scatter_at_1_then_2*
		 pow(atten_sc1_to_sc2,total_cross_section_scatter_at_1-1)*
		 pow(atten_to_detB,total_cross_section_relative_to_511keV(new_energy_scatter_at_1_then_2)-1)				 
		 +
		 emiss_to_detB*
		 dif_cross_section_sc1_sc2_B*
		 dif_cross_section_B_sc2_sc1_A*
//		 total_cross_section_scatter_at_1/total_cross_section_scatter_at_2*
		 detection_efficiency_scatter_at_2_then_1*
		 pow(atten_sc1_to_sc2,total_cross_section_scatter_at_2-1)*
		 pow(atten_to_detA,total_cross_section_relative_to_511keV(new_energy_scatter_at_2_then_1)-1)
		 )
		 /(rA_squared*rB_squared)
		 *atten_to_detB
		 *atten_to_detA
		 *atten_sc1_to_sc2
		 *scatter_point_1_mu
		 *scatter_point_2_mu;

	    const CartesianCoordinate3D<float> 
	      detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	    const CartesianCoordinate3D<float> 
	      detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	    const float cos_incident_angle_AS1 = 
	      cos_angle(scatter_point_1 - detector_coord_A,
			detA_to_ring_center) ;
	    const float cos_incident_angle_BS2 = 
	      cos_angle(scatter_point_2 - detector_coord_B,
			detB_to_ring_center) ;

	    return
	      scatter_ratio*cos_incident_angle_AS1*
	      //cos_incident_angle_AS1
	      //	      *cos_incident_angle_BS2*
	      cos_incident_angle_BS2 ;
	 }

 
	  if(split == 2){  // 1-ScatterBlue and 1-ScatterPink
	     const double scatter_ratio =
		 emiss_sc1_to_sc2*
		 dif_cross_section_A_sc1_sc2*
		 dif_cross_section_sc1_sc2_B
		 /detection_efficiency_no_scatter
		 *detection_efficiency_scatter_at_1*detection_efficiency_scatter_at_2*
		 pow(atten_to_detA,total_cross_section_scatter_at_1-1)*
		 pow(atten_to_detB,total_cross_section_scatter_at_2-1)
		 /(rA_squared*rB_squared)
		 *atten_to_detB
		 *atten_to_detA
		 *atten_sc1_to_sc2
		 *scatter_point_1_mu
		 *scatter_point_2_mu;

	    const CartesianCoordinate3D<float> 
	      detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	    const CartesianCoordinate3D<float> 
	      detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	    const float cos_incident_angle_AS1 = 
	      cos_angle(scatter_point_1 - detector_coord_A,
			detA_to_ring_center) ;
	    const float cos_incident_angle_BS2 = 
	      cos_angle(scatter_point_2 - detector_coord_B,
			detB_to_ring_center) ;

	    return
	      scatter_ratio*cos_incident_angle_AS1*
	      //cos_incident_angle_AS1
	      //	      *cos_incident_angle_BS2*
	      cos_incident_angle_BS2 ;
	 }



}
END_NAMESPACE_STIR
