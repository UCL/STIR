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

float att_estimate_for_no_scatter(	 
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const unsigned det_num_A, 
	  const unsigned det_num_B)
{		
	const CartesianCoordinate3D<float>& detector_coord_A =
		detection_points_vector[det_num_A];
    const CartesianCoordinate3D<float>& detector_coord_B =
		detection_points_vector[det_num_B];		
	 
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
	float atten_detA_to_detB = -rescale*integral_scattpoint_det(
			image_as_density,
			detector_coord_A, 
			detector_coord_B);
    
	return atten_detA_to_detB;	
}

END_NAMESPACE_STIR
