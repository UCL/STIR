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
using namespace std;
START_NAMESPACE_STIR
static const float total_cross_section_511keV = 
ScatterEstimationByBin::
  total_cross_section(511.F); 

// for compatiblity with scatter_viewgram.cxx 
float
ScatterEstimationByBin::
scatter_estimate_for_all_scatter_points(const unsigned det_num_A, 
					const unsigned det_num_B)	
{
  double scatter_ratio_01 = 0;
  double scatter_ratio_11 = 0;
  double scatter_ratio_02 = 0;

 scatter_estimate_for_all_scatter_points(scatter_ratio_01,
					 scatter_ratio_11,
					 scatter_ratio_02,
					 det_num_A, 
					 det_num_B);

  return scatter_ratio_01 + scatter_ratio_11 + scatter_ratio_02;
}      


void
ScatterEstimationByBin::
    scatter_estimate_for_all_scatter_points(double& scatter_ratio_01,
					    double& scatter_ratio_11,
					    double& scatter_ratio_02,
					    const unsigned det_num_A, 
					    const unsigned det_num_B)
{

  scatter_ratio_01 = 0;
  scatter_ratio_11 = 0;
  scatter_ratio_02 = 0;


  // TODO: slightly dangerous to use a static here
  // it would give wrong results when the energy_thresholds are changed...
  static const float detection_efficiency_no_scatter =
    detection_efficiency(511.F) > 0 
    ? detection_efficiency(511.F)
    : (std::cerr << "Zero detection efficiency for 511. Will normalise to 1\n", 1.F);

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
						 scatter_point_num,
						 det_num_A, det_num_B);	

      if(scatter_level==2||scatter_level==12||scatter_level==120)
	for(std::size_t scatter_point_2_num =0;
	    scatter_point_2_num < scatt_points_vector.size() ;
	    ++scatter_point_2_num)			
	  {
	    if(scatter_point_2_num!=scatter_point_num)
	      scatter_estimate_for_two_scatter_points(
							scatter_ratio_11,
							scatter_ratio_02,
							scatter_point_2_num,
							scatter_point_num,
							det_num_A, det_num_B);
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
