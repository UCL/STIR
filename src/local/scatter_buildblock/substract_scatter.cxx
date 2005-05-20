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
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h" 
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/CPUTimer.h"
#include "stir/Viewgram.h"
#include <fstream>
#include <cstdio>
using namespace std;

START_NAMESPACE_STIR

void substract_scatter(
		ProjData& corrected_scatter_proj_data, 
		const shared_ptr<ProjData> & no_scatter_proj_data_sptr, 
		const shared_ptr<ProjData> & scatter_proj_data_sptr, 
		const ProjData& att_proj_data, 
		const float global_scatter_factor)
{	
	const ProjDataInfoCylindricalNoArcCorr &proj_data_info = 
		dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&> 
		(*att_proj_data.get_proj_data_info_ptr());

    Bin bin;
	
	/////////////////// SCALING TIME /////////////////	
	CPUTimer bin_timer;
	int bin_counter = 0;
	bin_timer.start();
	int axial_bins = 0 ;
	for (bin.segment_num()=proj_data_info.get_min_segment_num();
	bin.segment_num()<=proj_data_info.get_max_segment_num();
	++bin.segment_num())	
		axial_bins += proj_data_info.get_num_axial_poss(bin.segment_num());	
    const int total_bins = proj_data_info.get_num_views() * axial_bins *
		proj_data_info.get_num_tangential_poss()	;
	/////////////////// end SCALING TIME /////////////////

	/* Currently, proj_data_info.find_cartesian_coordinates_of_detection() returns
	   coordinate in a coordinate system where z=0 in the first ring of the scanner.
	   We want to shift this to a coordinate system where z=0 in the middle 
	   of the scanner.
	   We can use get_m() as that uses the 'middle of the scanner' system.
	   (sorry)
	   */
	const CartesianCoordinate3D<float>  
			shift_detector_coordinates_to_origin(proj_data_info.get_m(Bin(0,0,0,0)),
				                                 0,
												 0);
	float total_outside_scatter = 0, total_outside_no_scatter =0, global_scale_factor = 0;

	for (bin.segment_num()=proj_data_info.get_min_segment_num();
	bin.segment_num()<=proj_data_info.get_max_segment_num();
	++bin.segment_num())
		for (bin.view_num()=proj_data_info.get_min_view_num();
		bin.view_num()<=proj_data_info.get_max_view_num();
		++bin.view_num())
		{
  			Viewgram<float> scatter_viewgram = scatter_proj_data_sptr->get_viewgram(
				bin.view_num(),bin.segment_num(),0);
			Viewgram<float> no_scatter_viewgram = no_scatter_proj_data_sptr->get_viewgram(
				bin.view_num(),bin.segment_num(),0);
			Viewgram<float> att_viewgram = att_proj_data.get_viewgram(
				bin.view_num(),bin.segment_num(),0);
			Viewgram<float> corrected_viewgram =
				corrected_scatter_proj_data.get_empty_viewgram(bin.view_num(), bin.segment_num());
						
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
					corrected_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]=
						no_scatter_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]-
						global_scatter_factor*scatter_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]*att_viewgram[bin.axial_pos_num()][bin.tangential_pos_num()];
					
					++bin_counter;
				}			    
					corrected_scatter_proj_data.set_viewgram(corrected_viewgram);

			
				/////////////////// SCATTER ESTIMATION TIME /////////////////												 						
				{				
					static double previous_timer = 0 ;		
					static int previous_bin_count = 0 ;
					cerr << bin_counter << " bins  Total time elapsed "
						 << bin_timer.value() << " sec \tTime remaining about "
						 << (bin_timer.value()-previous_timer)*(total_bins-bin_counter)
						/(bin_counter-previous_bin_count)/60						
						<< " minutes\n";				
					previous_timer = bin_timer.value() ;
					previous_bin_count = bin_counter ;
				}						
				/////////////////// end SCATTER ESTIMATION TIME /////////////////
		}
		bin_timer.stop();		
}
	END_NAMESPACE_STIR
		
