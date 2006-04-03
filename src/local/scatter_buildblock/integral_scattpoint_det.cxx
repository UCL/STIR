//
// $Id$
//
/*!
\file
\ingroup scatter
\brief Implementations of functions defined in Scatter.h
Function calculates the integral along LOR in an image (attenuation or emission). 
(From scatter point to detector coordinate)

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  		  
		$Date$
		$Revision$
		
		  Copyright (C) 2004- $Date$, Hammersmith Imanet
		  See STIR/LICENSE.txt for details
*/
#include "local/stir/ScatterEstimationByBin.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
START_NAMESPACE_STIR

float 
ScatterEstimationByBin::
integral_scattpoint_det (const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
							   const CartesianCoordinate3D<float>& scatter_point, 
							   const CartesianCoordinate3D<float>& detector_coord)
{				
	const VoxelsOnCartesianGrid<float>& image =
			dynamic_cast<const VoxelsOnCartesianGrid<float>& >(discretised_image);
	const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
	
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
					we_have_been_within_the_image = true;
					sum += image[coords] * element_ptr->get_value();				
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
}						   
END_NAMESPACE_STIR

