//
// $Id$
//
/*		
  Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup scatter
\brief Implementation of integration function

Function calculates the integral along LOR in an emission image 
(From scatter point to detector coordinate) taking solid angles into account.

  \author Nikolaos Dikaios
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  		  
  $Date$
  $Revision$
*/

#include "local/stir/ScatterEstimationByBin.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
START_NAMESPACE_STIR

float
ScatterEstimationByBin::
integral_over_activity_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point, 
						     const CartesianCoordinate3D<float>& detector_coord)
{
  if (!this->use_solid_angle_for_points)
  {
  const CartesianCoordinate3D<float> dist_vector = scatter_point - detector_coord ;

  const float dist_sp1_sp2_squared = norm_squared(dist_vector);

  const float solid_angle_factor = 
    std::min(static_cast<float>(_PI/2), 1.F  / dist_sp1_sp2_squared) ;
 
  return
    solid_angle_factor *
    integral_between_2_points(*activity_image_sptr,
			      scatter_point,
			      detector_coord);
  }
  else
  {
  const DiscretisedDensityOnCartesianGrid<3,float>& image =
  dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>& >
  (*activity_image_sptr);
				
 
  const CartesianCoordinate3D<float> voxel_size = image.get_grid_spacing();
  
	CartesianCoordinate3D<float>  origin = 
	  image.get_origin();
	const float z_to_middle =
	  (image.get_max_index() + image.get_min_index())*voxel_size.z()/2.F;
	origin.z() -= z_to_middle;
	
	ProjMatrixElemsForOneBin lor;

	RayTraceVoxelsOnCartesianGrid(lor, 
				      (scatter_point-origin)/voxel_size,  // should be in voxel units
				      (detector_coord-origin)/voxel_size, // should be in voxel units
				      voxel_size, //should be in mm
#ifdef NEWSCALE
				      1.F // normalise to mm
#else
				      1/voxel_size.x() // normalise to some kind of 'pixel units'
#endif
				      );
	lor.sort();
	float sum = 0;  // add up values along LOR
	{     
			ProjMatrixElemsForOneBin::iterator element_ptr =lor.begin() ;
			bool we_have_been_within_the_image = false;
			while (element_ptr != lor.end())
			  {
			 	const BasicCoordinate<3,int> coords = element_ptr->get_coords();			
				if (coords[1] >= image.get_min_index() && 
				    coords[1] <= image.get_max_index() &&
				    coords[2] >= image[coords[1]].get_min_index() && 
				    coords[2] <= image[coords[1]].get_max_index() &&
				    coords[3] >= image[coords[1]][coords[2]].get_min_index() && 
				    coords[3] <= image[coords[1]][coords[2]].get_max_index())
				  {
				    if (image[coords] == 0)
				      {
					++element_ptr;		
					continue;
				      }
				  we_have_been_within_the_image = true;
					
				  const CartesianCoordinate3D<float> emis_point=
				    voxel_size*BasicCoordinate<3,float>(coords) + origin;
				  
				  const float emis_to_det_solid_angle_factor = 
				    compute_emis_to_det_points_solid_angle_factor(emis_point,detector_coord) ;
				  float emis_sum = image[coords] * element_ptr->get_value();
					
			  
				  sum += emis_sum*emis_to_det_solid_angle_factor ;				
				  }
				else if (we_have_been_within_the_image)
				  {
				    // we jump out of the loop as we are now at the other side of 
				    // the image
				    //					break; 
				  }
				++element_ptr;		
			  }	      
		}  		
		return sum;	
  } // use_points
}						   
END_NAMESPACE_STIR

