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
#include "stir/CPUTimer.h"
#include "stir/Viewgram.h"
#include <fstream>

using namespace std;

START_NAMESPACE_STIR

void scatter_viewgram( 
					  ProjData& proj_data,
					  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					  const int scatt_points, const float att_threshold)
{	
	
	const ProjDataInfoCylindricalNoArcCorr &proj_data_info = 
		dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&> 
		(*proj_data.get_proj_data_info_ptr());
	
	std::vector<CartesianCoordinate3D<float> > scatt_points_vector =  
		sample_scatter_points(image_as_density,scatt_points,att_threshold);
	
	CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
    Bin bin;
	
	/////////////////// SCATTER ESTIMATION TIME /////////////////	
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
	/////////////////// end SCATTER ESTIMATION TIME /////////////////
	
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
					bin.set_bin_value(
						scatter_estimate_for_all_scatter_points(
						image_as_activity,
						image_as_density,
						scatt_points_vector, 
						detector_coord_A, 
						detector_coord_B));

					viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
						bin.get_bin_value();
					
					++bin_counter;
				}			    	
				proj_data.set_viewgram(viewgram);
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
		cerr << "Total Scatter Points : " << scatt_points_vector.size() << endl;			
			fstream mystream("statistics.txt", ios::out | ios::app); //output file //
				if(!mystream)    
					error("Cannot open text file.\n") ;					
				mystream  << "\n\t ********* NEW STATISTIC DATA *********\n" 
				 		<< "\nTotal bins: " << bin_counter 			      				
						<< "\tTotal minutes elapsed: "				  
						<< bin_timer.value()/60 
						<< "\nTotal Scatter Points : " << scatt_points_vector.size() << endl;
				bin_timer.stop();
				mystream.close();
	}
	END_NAMESPACE_STIR
		
