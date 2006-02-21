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
#include <fstream>
#include <ctime>
using namespace std;
START_NAMESPACE_STIR

static inline float random_point(const float low, const float high){
	/* returns a pseudo random number which holds in the bounds low and high */
	const float result= (rand()*(high-low))/RAND_MAX  + low;
	assert(low <= result);
	assert(high >= result);
	return result;
}

void
sample_scatter_points(const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
					  int & scatt_points, 
					  const float att_threshold,
					  const bool random)
{ 
	const DiscretisedDensityOnCartesianGrid<3,float>* attenuation_map_cartesian_ptr=
		&attenuation_map;   
	if (attenuation_map_cartesian_ptr == 0)
		warning("Didn't take an attenuation map as input");
	
	BasicCoordinate<3,int> min_index, max_index ;
	CartesianCoordinate3D<int> coord;
	
	if(!attenuation_map_cartesian_ptr->get_regular_range(min_index, max_index))
		error("scatter points sampling works only on regular ranges, at the moment\n");    
	const VoxelsOnCartesianGrid<float>& image =
		dynamic_cast<const VoxelsOnCartesianGrid<float>&>(attenuation_map);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size(); 	
	CartesianCoordinate3D<float>  origin = image.get_origin();
	// shift origin such that we refer to the middle of the scanner
	const float z_to_middle =
		(image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;
	origin.z() -= z_to_middle;

	if(random)
	{ // Initialize Pseudo Random Number generator using time  
				srand((unsigned)time( NULL ));
	}
	scatt_points_vector.resize(0); // make sure we don't keep scatter points from a previous run
	scatt_points_vector.reserve(std::min(1000, scatt_points));  

	// coord[] is in voxels units       
	for(coord[1]=min_index[1];coord[1]<=max_index[1];++coord[1])
		for(coord[2]=min_index[2];coord[2]<=max_index[2];++coord[2])
			for(coord[3]=min_index[3];coord[3]<=max_index[3];++coord[3])   
				if(attenuation_map[coord] >= att_threshold)
				{
					ScatterPoint scatter_point;					
					scatter_point.coord = convert_int_to_float(coord);
					if (random)
						scatter_point.coord +=
						CartesianCoordinate3D<float>(random_point(-.5,.5),
						random_point(-.5,.5),
						random_point(-.5,.5));
					scatter_point.coord =
						voxel_size*scatter_point.coord + origin;
					scatter_point.mu_value = attenuation_map[coord];
					scatt_points_vector.push_back(scatter_point);
				}					
	scatt_points = scatt_points_vector.size(); //this is the number of total scatt_points that is a refernece to the call
}
END_NAMESPACE_STIR 
