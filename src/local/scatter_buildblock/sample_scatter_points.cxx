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

std::vector<CartesianCoordinate3D<float> > 
sample_scatter_points(const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
					  int & max_scatt_points, 
					  float att_threshold)
{ 
   const DiscretisedDensityOnCartesianGrid<3,float>* attenuation_map_cartesian_ptr=
	     &attenuation_map;   
   if (attenuation_map_cartesian_ptr == 0)
	   warning("Didn't take an attenuation map as input");

      BasicCoordinate<3,int> min_index, max_index ;
   CartesianCoordinate3D<int> coord;
   
   if(!attenuation_map_cartesian_ptr->get_regular_range(min_index, max_index))
	   error("scatter points sampling works only on regular ranges, at the moment\n");
    
   int total_points=1;

   const VoxelsOnCartesianGrid<float>& image =
  dynamic_cast<const VoxelsOnCartesianGrid<float>&>(attenuation_map);
   const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size(); 

   std::vector<CartesianCoordinate3D<int> > points; 
     
   // coord[] is voxels units       
   for(coord[1]=min_index[1];coord[1]<=max_index[1];++coord[1])
	   for(coord[2]=min_index[2];coord[2]<=max_index[2];++coord[2])
		   for(coord[3]=min_index[3];coord[3]<=max_index[3];++coord[3])
			   points.push_back(coord);
 
   std::random_shuffle(points.begin(),points.end()); 
   std::vector<CartesianCoordinate3D<int> >:: iterator current_iter;
   std::vector<CartesianCoordinate3D<float> > scatt_points;

   for(current_iter=points.begin(); 
       current_iter!=points.end() && total_points<=max_scatt_points;
	   ++current_iter)			   
		   if(attenuation_map[(*current_iter)]>=att_threshold)
		   {			
			   scatt_points.push_back(voxel_size*convert_int_to_float(*current_iter));
			   ++total_points;
		   }	
    if (total_points <= max_scatt_points) 
	{
		warning("The att_threshold or the max_scatt_points are set too high!");	
		max_scatt_points = total_points-1; 
	}	
	// in mm units 
    return scatt_points;
}

END_NAMESPACE_STIR 
