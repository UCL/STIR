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
#include <cmath>

using namespace std;

START_NAMESPACE_STIR

static const float total_cross_section_511keV = 
  total_cross_section(511.); 

float scatter_estimate_for_one_scatter_point(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const std::size_t scatter_point_num, 
	  const unsigned det_num_A, 
	  const unsigned det_num_B)
{	
	// TODO hard-wired for now
//	static const lower_energy_threshold = 350;
//	static const upper_energy_threshold = 650;
	static const float lower_energy_threshold = 375;
	static const float upper_energy_threshold = 600;

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
	// Hence, the cross_section is identical for both cases as well.
	const float dif_cross_section =
		dif_cross_section_511keV(costheta); 
	const float new_energy =
		energy_after_scatter_511keV(costheta);

	const CartesianCoordinate3D<float> detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
	const CartesianCoordinate3D<float> detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
	const float cos_incident_angle_A = 
		cos_angle(scatter_point - detector_coord_A,
		          detA_to_ring_center) ;
	const float cos_incident_angle_B = 
		cos_angle(scatter_point - detector_coord_B,
		          detB_to_ring_center) ;

	// TODO: slightly dangerous to use a static here
	// it would give wrong results when the energy_thresholds are changed...
	static const float detection_efficiency_no_scatter =
		detection_efficiency_BGO(lower_energy_threshold,
                                 upper_energy_threshold,
                                 512);
	const float detection_efficiency_scatter =
		detection_efficiency_BGO(lower_energy_threshold,
                                 upper_energy_threshold,
                                 new_energy);

	if (detection_efficiency_scatter==0)
		return 0;

	const float
		emiss_to_detA = cached_factors(
	                             image_as_activity,
	                             scatter_point_num, 
                                 det_num_A
								 , act_image_type
							     );
    const float
	  emiss_to_detB = cached_factors(
		                         image_as_activity,
								 scatter_point_num, 
								 det_num_B
								 ,act_image_type
							     );
	if (emiss_to_detA==0 && emiss_to_detB==0)
		return 0;

    // TODO in principle, the scattered photon should have different attenuation
    const float 
		atten_to_detA = cached_factors(
	                             image_as_density,
	                             scatter_point_num, 
                                 det_num_A
								 , att_image_type
								 );
	const float
		atten_to_detB = cached_factors(
	                             image_as_density,
	                             scatter_point_num, 
                                 det_num_B
						    	 , att_image_type
								 );

	const VoxelsOnCartesianGrid<float>& image =
		dynamic_cast<const VoxelsOnCartesianGrid<float>&>(image_as_density);
	
	
	const float rA=norm(scatter_point-detector_coord_A);
	const float rB=norm(scatter_point-detector_coord_B);
	
	const float scatter_point_mu=
		scatt_points_vector[scatter_point_num].mu_value;
#ifndef NDEBUG
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	CartesianCoordinate3D<float>  origin = 
		image.get_origin();
	origin.z() -= 
		(image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;

	assert(scatter_point_mu==
		image[round((scatter_point-origin)/voxel_size)]);
#endif

	return 
		(emiss_to_detA + emiss_to_detB)
		/(rB*rB*rA*rA)
		*dif_cross_section
		*atten_to_detA
		*atten_to_detB
		*scatter_point_mu
		*detection_efficiency_no_scatter
		*detection_efficiency_scatter
//		*cos_incident_angle_A*cos_incident_angle_A
//		*cos_incident_angle_B*cos_incident_angle_B 
		*exp(-total_cross_section(new_energy)/total_cross_section_511keV)
        /total_cross_section_511keV
		;	
}

END_NAMESPACE_STIR
