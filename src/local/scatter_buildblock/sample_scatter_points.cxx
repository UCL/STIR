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

template <class elemT>
std::vector<CartesianCoordinate3D<int> > 
sample_scatter_points(const DiscretisedDensityOnCartesianGrid<3,elemT>& attenuation_map,
					  int & max_scatt_points, 
					  elemT att_threshold)
{ 
   const DiscretisedDensityOnCartesianGrid <3,elemT>* attenuation_map_cartesian_ptr=
	     &attenuation_map;   
   if (attenuation_map_cartesian_ptr == 0)
   {
      warning("Didn't take an attenuation map as input");
      return EXIT_FAILURE;
   }

   BasicCoordinate<3,int> min_index, max_index ;
   CartesianCoordinate3D<int> scatt_coord, att_map_size;
   
   if(!attenuation_map_cartesian_ptr->get_regular_range(min_index, max_index))
	   error("scatter points sampling works only on regular ranges, at the moment\n");
    
   int total_points=1;


   std::vector<CartesianCoordinate3D<int> > points, scatt_points ;
    
   CartesianCoordinate3D<int> coord;
   for(coord[1]=min_index[1];coord[1]<=max_index[1];++coord[1])
	   for(coord[2]=min_index[2];coord[2]<=max_index[2];++coord[2])
		   for(coord[3]=min_index[3];coord[3]<=max_index[3];++coord[3])
			   points.push_back(coord);
 
   std::random_shuffle(points.begin(),points.end()); 
   std::vector<CartesianCoordinate3D<int> >:: iterator current_iter;

   for(current_iter=points.begin(); 
       current_iter!=points.end() && total_points<=max_scatt_points;
	   ++current_iter)	 			  	      
		   
		   if(attenuation_map[*current_iter]>=att_threshold)
		   {
			   scatt_points.push_back(*current_iter);
			   ++total_points;
		   }	
    if (total_points <= max_scatt_points) 
	{
		warning("The att_threshold or the max_scatt_points are set too high!");	
		max_scatt_points = total_points-1; 
	}	
    return scatt_points;
}
/***************************************************
                 instantiations
***************************************************/
template
std::vector<CartesianCoordinate3D<int> > 
sample_scatter_points<>(const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
					  int & max_scatt_points, 
					  float att_threshold) ;

END_NAMESPACE_STIR 
