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
	mystream << " Output_proj_data_filename is:" << argv[4]
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
		<< "\nThreshold was set to: " << given_attenuation_threshold
		<< "\nScatter Points are taken all above the threshold";
		
	if (random)
		mystream << "\nand have picked randomly ";
	else
	mystream << "and have picked in the center of the Voxel";
    mystream <<"\n\n\t ****************** END ****************\n";
}

void writing_time(const int simulation_time, const int scatt_points_vector_size)
{
		{
			fstream mystream("statistics.txt", ios::out | ios::app); //output file //
			if(!mystream)    
				warning("Cannot open statistics.txt file.\n") ;
			else
			{
				mystream  << "\n\t ********* NEW STATISTIC DATA *********\n" 
			   	    << "\tTotal simulation time elapsed: "				  
					<<   simulation_time/60 
					<< "\nTotal Scatter Points : " << scatt_points_vector_size << endl;
			}
		}
}

END_NAMESPACE_STIR