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

	float scatter_estimate_for_all_scatter_points(
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
    const std::vector<CartesianCoordinate3D<float> >& scatter_points_vector, 
	const CartesianCoordinate3D<float>& detector_coord_A, 
	const CartesianCoordinate3D<float>& detector_coord_B)
{	
	float scatter_ratio = 0; 
	
	for(std::vector<CartesianCoordinate3D<float> >:: const_iterator 
		current_iter_point = scatter_points_vector.begin();
	    current_iter_point != scatter_points_vector.end() ;
	  ++current_iter_point)
	  {
		  scatter_ratio +=
			  scatter_estimate_for_one_scatter_point(
			  image_as_activity, image_as_density, 
			  *current_iter_point,
			  detector_coord_A, detector_coord_B);
	  }
    
	return scatter_ratio;
}


END_NAMESPACE_STIR
