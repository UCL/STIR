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
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h" 
#include "stir/Viewgram.h"

using namespace std;

START_NAMESPACE_STIR

void scatter_viewgram( 
	ProjData& proj_data,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
    int max_scatt_points, const float att_threshold)
	{	
		  	  
	const ProjDataInfoCylindricalNoArcCorr &proj_data_info = 
		dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&> 
		(*proj_data.get_proj_data_info_ptr());
		
			
			//proj_data.get_proj_data_info_ptr());

	std::vector<CartesianCoordinate3D<float> > scatter_points_vector =  
		sample_scatter_points(image_as_density,max_scatt_points,att_threshold);
	
	CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
    Bin bin;
	
	for (bin.segment_num()=proj_data_info.get_min_segment_num();
	bin.segment_num()<=proj_data_info.get_max_segment_num();
	++bin.segment_num())
		for (bin.view_num()=proj_data_info.get_min_view_num();
		bin.view_num()<=proj_data_info.get_max_view_num();
		++bin.view_num())
			{
			Viewgram<float> viewgram =
				proj_data.get_empty_viewgram(bin.view_num(), bin.segment_num());
			
			for (bin.axial_pos_num()=
				proj_data_info.get_min_axial_pos_num(bin.segment_num());
			bin.axial_pos_num()<=proj_data_info.get_max_axial_pos_num(bin.segment_num());
			++bin.axial_pos_num())
				for (bin.tangential_pos_num()=
					proj_data_info.get_min_tangential_pos_num();
				bin.tangential_pos_num()<=
					proj_data_info.get_max_tangential_pos_num();
				++bin.tangential_pos_num())
					{  
					// have now all bin coordinates
					proj_data_info.find_cartesian_coordinates_of_detection(
						detector_coord_A,detector_coord_B,bin);
					
					bin.set_bin_value(scatter_estimate_for_all_scatter_points(
						image_as_activity,
						image_as_density,
						scatter_points_vector, 
						detector_coord_A, 
						detector_coord_B));
					
					viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
						bin.get_bin_value();
					
					}			    	
				proj_data.set_viewgram(viewgram);
			}
		
	}
END_NAMESPACE_STIR

