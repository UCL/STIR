//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief A collection of functions to write the statistic after the measure of the single scatter component
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$

 */
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Viewgram.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "local/stir/Scatter.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
using std::setw;
#endif

using namespace std;
START_NAMESPACE_STIR

    /****************************************************/ 
	/* WRITING THE STATISTIC DATA IN THE STATISTIC FILE */
    /****************************************************/ 

void writing_log(const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
				 const DiscretisedDensityOnCartesianGrid<3,float>& density_image,
				 const ProjDataInfoCylindricalNoArcCorr* proj_data_info_ptr,
				 const float given_attenuation_threshold,
				 const int total_scatt_points, 
				 const float lower_energy_threshold, 
				 const float upper_energy_threshold,
				 const bool use_cache,
				 const bool random, const char *argv[])
{	
	Bin bin;
	int axial_bins = 0 ;
	for (bin.segment_num()=proj_data_info_ptr->get_min_segment_num();
	bin.segment_num()<=proj_data_info_ptr->get_max_segment_num();
	++bin.segment_num())	
		axial_bins += proj_data_info_ptr->get_num_axial_poss(bin.segment_num());	
    const int total_bins = proj_data_info_ptr->get_num_views() * axial_bins *
		proj_data_info_ptr->get_num_tangential_poss()	;	
	/*
	cerr << "\n Total bins : " << total_bins << " = " 
		 << proj_data_info_ptr->get_num_views() 
		 << " view_bins * " 
		 << axial_bins << " axial_bins * "
		 << proj_data_info_ptr->get_num_tangential_poss() 
		 << " tangential_bins\n"   ;		
    */
	fstream mystream("statistics.txt", ios::out | ios::app); //output file //
	if(!mystream)    
		warning("Cannot open text file.\n") ;	              
	mystream << "Output_proj_data_filename is:" << argv[4]
		<< "\nActivity image: " << argv[1]
		<< "\nwith SIZE: " 
		<< activity_image.get_z_size() << " * " 
		<< activity_image.get_y_size() << " * " 
		<< activity_image.get_x_size()
		<< "\nAttenuation image: " << argv[2]
		<< "\nwith SIZE: " 
		<< density_image.get_z_size() << " * "
		<< density_image.get_y_size() << " * " 
		<< density_image.get_x_size()
		<< "\n\nTemplate proj_data: " << argv[3]
		<< "\nwith total bins : " << total_bins << " = " 
		<< proj_data_info_ptr->get_num_views() 		 
		<< " view_bins * " 
		<< axial_bins << " axial_bins * "
		<< proj_data_info_ptr->get_num_tangential_poss() 
		<< " tangential_bins\n"  
		<< "\n - The energy window was set to: [" 
		<< lower_energy_threshold << "," << upper_energy_threshold 
		<< "]\n - Threshold was set to: " << given_attenuation_threshold
		<< "\n - Scatter Points are taken all above the threshold";		
	if (random)
		mystream << " and have picked randomly\n";
	if (!random)
		mystream << " and have picked in the center of the Voxel\n";
	if(use_cache)
		mystream << " - Use of caching\n";
	if(!use_cache)
		mystream << " - No use of caching for SS - Use of caching for the LoRs in DS\n";
    mystream <<"\n\n\t ****************** END ****************\n\n\n\n\n\a";
}
void writing_time(const double simulation_time, 
				  const int scatt_points_vector_size, 
				  const int scatter_level,
				  const float total_scatter)
{
	{
		fstream mystream("statistics.txt", ios::out | ios::app); //output file //
		if(!mystream)    
			warning("Cannot open statistics.txt file.\n") ;
		else
		{
			mystream  << "\n  ****** NEW STATISTIC DATA FOR ";
			if(scatter_level==2)
				mystream << "ONLY DOUBLE ";
			if(scatter_level==1)
				mystream << "ONLY SINGLE ";
			if(scatter_level==0)
				mystream << "NO ";
			if(scatter_level==10)
				mystream << "SINGLE and NO ";
			if(scatter_level==12)
				mystream << "SINGLE and DOUBLE ";
			if(scatter_level==120)
				mystream << "SINGLE, DOUBLE and NO";
		
			mystream  << "SCATTER SIMULATION ******\n\n"
				<< "Total simulation time elapsed: "				  
				<<   simulation_time/60 
				<< "\nTotal Scatter Points : " << scatt_points_vector_size 
				<< "\nScatter Estimation : " << total_scatter << endl;
		}
	}
}
END_NAMESPACE_STIR

