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

float scatter_estimate_for_one_scatter_point(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const CartesianCoordinate3D<float>& scatter_point, 
	  const CartesianCoordinate3D<float>& detector_coord_A, 
	  const CartesianCoordinate3D<float>& detector_coord_B)
{	

    const float 
		atten_integral_to_detA = integral_scattpoint_det(
	                               image_as_density,
	                               scatter_point, 
                                   detector_coord_A),
		emiss_integral_to_detA = integral_scattpoint_det(
	                               image_as_activity,
	                               scatter_point, 
                                   detector_coord_A),
		atten_integral_to_detB = integral_scattpoint_det(
	                               image_as_density,
	                               scatter_point, 
                                   detector_coord_B),
		emiss_integral_to_detB = integral_scattpoint_det(
	                               image_as_activity,
								   scatter_point, 
								   detector_coord_B);

	const VoxelsOnCartesianGrid<float>& image =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(image_as_density);

    const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();

  float rA=two_points_distance(scatter_point,detector_coord_A);
	float rB=two_points_distance(scatter_point,detector_coord_B);

	float scatter_point_mue=
		image[convert_float_to_int(scatter_point/voxel_size)];


	const float scatter_point_to_detA_ratio=
    emiss_integral_to_detA*exp(-atten_integral_to_detA)*exp(-atten_integral_to_detB) 
	*scatter_point_mue*dif_cross_section(scatter_point,detector_coord_A,detector_coord_B,511)
	/(total_cross_section(511)*rA*rA); 
	// ( in v2 ->* efficiencies, for the E')

	const float scatter_point_to_detB_ratio=
	emiss_integral_to_detB*exp(-atten_integral_to_detB)*exp(-atten_integral_to_detA) 
	*scatter_point_mue*dif_cross_section(scatter_point,detector_coord_B,detector_coord_A,511)
	/(total_cross_section(511)*rB*rB); 
	// ( in v2 ->* efficiencies, for the E')
	
	return scatter_point_to_detA_ratio+scatter_point_to_detB_ratio;
}

END_NAMESPACE_STIR
