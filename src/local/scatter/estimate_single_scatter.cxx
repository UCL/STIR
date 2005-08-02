//
// $Id$
//
/*!
\file
\ingroup utilities
\brief   

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	  estimate_single_scatter [activity_image]
	                          [attenuation_image]
							  [proj_data_filename]
							  [scatter_viewgram_filename]
							  [attenuation_threshold]
							  [lower_energy_threshold]
							  [upper_energy_threshold]
							  [maximum_scatter_points]
							  [random points]
							  [use_cache]
							  [scatter_level]
							  [resolution]
  	
	  Output: Viewgram with name scatter_viewgram_filename
              statistics.txt
	  \endcode
	  \param attenuation_threshold defaults to .05 cm^-1
	  \param lower_energy_threshold defaults to 350 keV
	  \param upper_energy_threshold defaults to 650 keV		  
	  \param maximum_scatter_points defaults to 1000: Input not activated	
	  \param random points defaults to true
	  \param use_cache defaults to true
	  \param scatter_level defaults to 1 (Single Scatter)
	  \param resolution defaults for BGO to 22%
*/
/*
Copyright (C) 2004- $Date$, Hammersmith Imanet
See STIR/LICENSE.txt for details
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "local/stir/Scatter.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
#endif

/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         
	USING_NAMESPACE_STIR
		using namespace std;
	if (argc< 7 || argc>14)
	{
	   cerr << "Usage:" << argv[0] << "\n"
			<< "\t[activity_image]\n"
			<< "\t[attenuation_image]\n"
			<< "\t[proj_data_filename]\n" 
			<< "\t[output_proj_data_filename]\n"
			<< "\t[attenuation_threshold]\n"
			<< "\t[lower_energy_threshold]\n"
			<< "\t[upper_energy_threshold]\n"
			<< "\t[maximum_scatter_points]\n"
			<< "\t[random points]\n"
			<< "\t[use_cache]\n"
			<< "\t[scatter_level]\n"
			<< "\t[resolution]\n\n"
			<< "\tattenuation_threshold defaults to .05 cm^-1\n"
			<< "\tlower_energy_threshold defaults to 350 keV\n"
			<< "\tupper_energy_threshold defaults to 650 keV\n"			
			<< "\tmaximum_scatter_points defaults to 1000\n" 
			<< "\trandom points defaults to true, use 0 to set to false\n"
			<< "\tuse_cache defaults to true, use 0 to set to false\n"
			<< "\tscatter_level defaults to 10 finds the no scatter + SSS sinogram\n"
			<< "\tscatter_level to 12 finds the SSS+DSS\n"
			<< "\tuse 1 for only SSS\n"
			<< "\t    2 for only DSS\n" 
			<< "\tor  0 for no scatter sinogram\n"
		        << "\tresolution defaults for BGO to 22%\n\n" ;
		return EXIT_FAILURE;            
	}      
	float attenuation_threshold = argc>=6 ? atof(argv[5]) : 0.05 ;
	const float lower_energy_threshold = argc>=7 ? atof(argv[6]) : 350 ;
	const float upper_energy_threshold = argc>=8 ? atof(argv[7]) : 650 ;
	int scatt_points = argc>=9 ? atoi(argv[8]) : 1000 ;
	bool random = true;
	if (argc>=10 && atoi(argv[9])==0)
		random = false;
	bool use_cache = true;
	if (argc>=11 && atoi(argv[10])==0)
		use_cache = false;
	const int scatter_level = argc>= 12 ? atoi(argv[11]) : 10 ;
	const float resolution = argc>=13 ? atoi(argv[12]) : 0.22 ;
	
	shared_ptr< DiscretisedDensity<3,float> >  
		activity_image_sptr= 
		DiscretisedDensity<3,float>::read_from_file(argv[1]), 
		density_image_sptr= 
		DiscretisedDensity<3,float>::read_from_file(argv[2]);
	
	warning("\nWARNING: Attenuation image data are supposed to be in units cm^-1\n"
		"\tReference: water has mu .096 cm^-1\n" 
		"\tMax in attenuation image: %g\n" ,
		density_image_sptr->find_max());
	shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);  
	const ProjDataInfoCylindricalNoArcCorr* proj_data_info_ptr =
		dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(
		template_proj_data_sptr->get_proj_data_info_ptr());
	
	if (proj_data_info_ptr==0 || density_image_sptr==0 || activity_image_sptr==0)
		error("Check the input files\n");
	const DiscretisedDensityOnCartesianGrid<3,float>& activity_image = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>&  > 
		(*activity_image_sptr.get());
	const DiscretisedDensityOnCartesianGrid<3,float>& density_image = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>&  > 
		(*density_image_sptr.get());
	
    string output_proj_data_filename(argv[4]);    		
	ProjDataInterfile output_proj_data(proj_data_info_ptr->clone(),output_proj_data_filename);
	
	cout << "\nwriting the single scatter contribution into the file: " 
		 << output_proj_data_filename <<".s\n"
		 << "and the statistics into the statistics.txt\n"
		 << "***********************************************************\n"
		 << "The simulation has started...\n";	
	
	scatter_viewgram(output_proj_data,
		activity_image, density_image,
		scatt_points,attenuation_threshold,
			 lower_energy_threshold,upper_energy_threshold,resolution,
		use_cache,scatter_level,random);  

	writing_log(activity_image,
		density_image,
		proj_data_info_ptr,
		attenuation_threshold,
		scatt_points,
		lower_energy_threshold,
		upper_energy_threshold,
	        resolution,
		use_cache,
		random,argv);

	return EXIT_SUCCESS;
}                 
