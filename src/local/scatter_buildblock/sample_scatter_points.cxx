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

// Function that will be in the BasicCoordinate Class
BasicCoordinate<3,float> convert_int_to_float(const BasicCoordinate<3,int>& cint)
	{	  
	  BasicCoordinate<3,float> cfloat;
	  cfloat[1]=(float)cint[1];
	  cfloat[2]=(float)cint[2];
	  cfloat[3]=(float)cint[3];
	  return cfloat;
	}

std::vector<CartesianCoordinate3D<float> > 
sample_scatter_points(const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
					  int & max_scatt_points, 
					  float att_threshold)
{ 
   const DiscretisedDensityOnCartesianGrid<3,float>* attenuation_map_cartesian_ptr=
	     &attenuation_map;   
   if (attenuation_map_cartesian_ptr == 0)
   {
      warning("Didn't take an attenuation map as input");
      return EXIT_FAILURE;
   }

   BasicCoordinate<3,int> min_index, max_index ;
   CartesianCoordinate3D<int> coord;
   
   if(!attenuation_map_cartesian_ptr->get_regular_range(min_index, max_index))
	   error("scatter points sampling works only on regular ranges, at the moment\n");
    
   int total_points=1;


   std::vector<CartesianCoordinate3D<int> > points; 
            
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
			   scatt_points.push_back(convert_int_to_float(*current_iter));
			   ++total_points;
		   }	
    if (total_points <= max_scatt_points) 
	{
		warning("The att_threshold or the max_scatt_points are set too high!");	
		max_scatt_points = total_points-1; 
	}	
    return scatt_points;
}

END_NAMESPACE_STIR 
