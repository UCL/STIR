//
// $Id$
//
/*!
\file
\ingroup utilities
\brief   

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	   correct_for_scatter [activity_image]
	                       [attenuation_image]
						   [scatter_viewgram]
						   [corrected_scatter_filename]
						   [attenuation_threshold]
						   [global_scale_factor]
							
	  Output: Viewgram with name corrected_scatter_filename
              
	  \endcode
	  \param attenuation_threshold defaults to .05 cm^-1	  
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
	if (argc< 5 || argc>7)
	{
	   cerr << "Usage:" << argv[0] << "\n"
			<< "\t[attenuation_image]\n"
			<< "\t[emission_projdata]\n"
			<< "\t[scatter_projdata]\n" 
			<< "\t[corrected_scatter_filename]\n"
			<< "\t[attenuation_threshold]\n"
			<< "\t[global_scale_factor]\n"
			<< "\tattenuation_threshold defaults to .05 cm^-1\n"
			<< "\tusing defaults to 0 for global scaling"
			<< "\tother value for manual scaling based AttProjData(EmProjData-scale*ScatterProjData)";
		return EXIT_FAILURE;            
	}      
	const float attenuation_threshold = argc>=6 ? atof(argv[5]) : 0.05 ;
	float global_scale_factor = argc>=7 ? atof(argv[6]) : 0 ; 
	
	shared_ptr< DiscretisedDensity<3,float> >  	
		density_image_sptr= 
		DiscretisedDensity<3,float>::read_from_file(argv[1]);
	
	warning("\nWARNING: Attenuation image data are supposed to be in units cm^-1\n"
		"\tReference: water has mu .096 cm^-1\n" 
		"\tMax in attenuation image: %g\n" ,
		density_image_sptr->find_max());

	shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[2]);  
	const ProjDataInfoCylindricalNoArcCorr* proj_data_info_ptr =
		dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(
		template_proj_data_sptr->get_proj_data_info_ptr());
	
	if (proj_data_info_ptr==0 || density_image_sptr==0)
		error("Check the input files\n");
	const DiscretisedDensityOnCartesianGrid<3,float>& density_image = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>&  > 
		(*density_image_sptr.get());
	
	string corrected_scatter_filename(argv[4]);    			
	ProjDataInterfile corrected_scatter_proj_data(proj_data_info_ptr->clone(), corrected_scatter_filename);
	
	string att_proj_data_filename("att_proj_data");
	ProjDataInterfile att_proj_data(proj_data_info_ptr->clone(), att_proj_data_filename);

	const shared_ptr<ProjData> no_scatter_proj_data_sptr = ProjData::read_from_file(argv[2]);  
	//const ProjDataInfo * projdata_info_ptr = 
    //(*no_scatter_proj_data_sptr).get_proj_data_info_ptr();  
	
	const shared_ptr<ProjData> scatter_proj_data_sptr = ProjData::read_from_file(argv[3]);   
	//const ProjDataInfo * projdata_info_ptr = 
    //(*scatter_proj_data_sptr).get_proj_data_info_ptr();
   
	estimate_att_viewgram(att_proj_data, density_image);
	
	if (global_scale_factor==0)
	global_scale_factor = estimate_scale_factor(
	    no_scatter_proj_data_sptr, 
		scatter_proj_data_sptr, 
		att_proj_data,
		attenuation_threshold) ;


	substract_scatter(corrected_scatter_proj_data, 
		no_scatter_proj_data_sptr,
		scatter_proj_data_sptr, 
		att_proj_data, 
		global_scale_factor) ;

	return EXIT_SUCCESS;
}                 
