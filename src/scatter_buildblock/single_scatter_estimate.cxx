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

// for compatiblity with scatter_viewgram.cxx 
float
 scatter_estimate_for_all_scatter_points(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,
	  const float resolution,		
	  const bool use_cache,
	  const bool use_sintheta, const bool use_polarization,
	  const int scatter_level)	
{	
  double scatter_ratio_01 = 0;
  double scatter_ratio_11 = 0;
  double scatter_ratio_02 = 0;
  const CartesianCoordinate3D<float> voxel_size = 
    image_as_density.get_grid_spacing();
  const float scatter_volume = voxel_size[1]*voxel_size[2]*voxel_size[3];

  scatter_estimate_for_all_scatter_points(
					  scatter_ratio_01,
					   scatter_ratio_11,
					  scatter_ratio_02,
					  image_as_activity,
					  image_as_density,
					  scatter_volume,
					  det_num_A, 
					  det_num_B,
					  lower_energy_threshold, 
					  upper_energy_threshold,
					  resolution,		
					  use_cache,
					  use_sintheta, use_polarization,
					  scatter_level);
  return scatter_ratio_01 + scatter_ratio_11 + scatter_ratio_02;
}      


void
 scatter_estimate_for_all_scatter_points(
					 double& scatter_ratio_01,
					 double& scatter_ratio_11,
					 double& scatter_ratio_02,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					 const float scatter_volume,
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,
	  const float resolution,		
	  const bool use_cache,
	  const bool use_sintheta,
	  const bool use_polarization,
	  const int scatter_level)	
{	
  scatter_ratio_01 = 0;
  scatter_ratio_11 = 0;
  scatter_ratio_02 = 0;

  // TODO: slightly dangerous to use a static here
  // it would give wrong results when the energy_thresholds are changed...
  static const float detection_efficiency_no_scatter =
    detection_efficiency(lower_energy_threshold,
			 upper_energy_threshold,
			 511.F,
			 511.F,
			 resolution);
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
		
  for(std::size_t scatter_point_num =0;
      scatter_point_num < scatt_points_vector.size();
      ++scatter_point_num)
    {	
      if(scatter_level==1||scatter_level==12||scatter_level==10||scatter_level==120)
	scatter_ratio_01 +=
	  scatter_estimate_for_one_scatter_point(
						 image_as_activity, image_as_density, 
						 scatter_point_num,
						 det_num_A, det_num_B,
						 lower_energy_threshold, 
						 upper_energy_threshold,
						 resolution,
						 use_cache, use_sintheta);	

      if(scatter_level==2||scatter_level==12||scatter_level==120)
	for(std::size_t scatter_point_2_num =0;
	    scatter_point_2_num < scatt_points_vector.size() ;
	    ++scatter_point_2_num)			
	  {
	    if(scatter_point_2_num!=scatter_point_num)
	      scatter_estimate_for_two_scatter_points(
							scatter_ratio_11,
							scatter_ratio_02,
							image_as_activity, image_as_density, 
							scatter_point_2_num,
							scatter_point_num,
							det_num_A, det_num_B,
							lower_energy_threshold, 
							upper_energy_threshold,
							resolution,
							use_cache, use_sintheta, use_polarization);
	  }
    }	

  // we will divide by the effiency of the detector pair for unscattered photons
  // (computed with the same detection model as used in the scatter code)
  // This way, the scatter estimate will correspond to a 'normalised' scatter estimate.

  // there is a scatter_volume factor for every scatter point, as the sum over scatter points
  // is an approximation for the integral over the scatter point.

  // the factors total_cross_section_511keV should probably be moved to the scatter_computation code

  //0.75 is due to the volume of the pyramid approximation!
  const double common_factor =
    0.75/2./_PI *
    rAB_squared*scatter_volume/total_cross_section_511keV
    /detection_efficiency_no_scatter/
    (cos_incident_angle_A*
     cos_incident_angle_B);

  scatter_ratio_01 *= common_factor;
  scatter_ratio_02 *= scatter_volume/total_cross_section_511keV * common_factor;
  scatter_ratio_11 *= scatter_volume/total_cross_section_511keV * common_factor;
}

END_NAMESPACE_STIR
