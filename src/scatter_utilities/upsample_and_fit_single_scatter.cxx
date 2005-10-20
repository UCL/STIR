//
// $Id$
//
/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief   

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
	
  \par Usage:
  see .cxx file  
\code
  \endcode
  						
  Output: Viewgram with name scaled_scatter_filename
              
  \param attenuation_threshold defaults to 1.01 (should be larger than 1)	  
*/

#include <iostream>
#include <fstream>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "local/stir/inverse_SSRB.h"
#include "local/stir/Scatter.h"
#include "local/stir/interpolate_projdata.h"
#include "stir/utilities.h"
#include "stir/IndexRange2D.h" 
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         
	USING_NAMESPACE_STIR
	if (argc< 5 || argc>9)
	{
	   std::cerr << "Usage:\n" << argv[0] << "\n"
		     << "\tattenuation_correction_factors\n"
		     << "\temission_projdata\n"
		     << "\tscatter_projdata\n" 
		     << "\toutput_filename\n"
		     << "\t[attenuation_threshold \n"
		     << "\t[correct_for_interleaving]\n"
		     << "\t[estimate_scale_factor]\n"
		     << "\t[mask_radius_in_mm]]]\n"
		     << "correct_for_interleaving default to 1\n"
		     << "attenuation_threshold defaults to 1.01\n" 
		     << "estimate_scale_factor defaults to 1 for scaling per sinogram\n"
		     << "0 for  no scaling, 2 for by viewgram\n"
		     << "If mask_radius_in_mm is not specified, will use all available data\n";
		return EXIT_FAILURE;            
	}      
	const float attenuation_threshold = argc>=6 ? atof(argv[5]) : 1.01 ;
	const bool remove_interleaving = argc>=7 ? atoi(argv[6]) : 1 ; 
	const int est_scale_factor_per_sino = argc>=8 ? atoi(argv[7]) : 1 ; 
	const float mask_radius_in_mm = argc>=9 ? atof(argv[8]) : -1 ; 

	shared_ptr< ProjData >  	
		attenuation_correct_factors_sptr= 
		ProjData::read_from_file(argv[1]);
	
	if (is_null_ptr(attenuation_correct_factors_sptr))
	  error("Check the attenuation_correct_factors file\n");

	const shared_ptr<ProjData> emission_proj_data_sptr = ProjData::read_from_file(argv[2]);  

	const shared_ptr<ProjData> scatter_proj_data_sptr = ProjData::read_from_file(argv[3]);   
	
	shared_ptr<ProjDataInfo> emission_proj_data_info_sptr =
		emission_proj_data_sptr->get_proj_data_info_ptr()->clone();
	
	string scaled_scatter_filename(argv[4]);    			
	ProjDataInterfile scaled_scatter_proj_data(emission_proj_data_info_sptr, scaled_scatter_filename);
	
	// interpolate scatter sinogram
	// first call interpolate_projdata to 'expand' segment 0 to appropriate size (i.e. same as emission data)
	// then call inverse_SSRB to generate oblique segments

	shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr =
	  emission_proj_data_sptr->get_proj_data_info_ptr()->clone();
	interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0);

	std::cout << "Interpolating scatter estimate to size of emission data" << std::endl;
	ProjDataInMemory interpolated_direct_scatter(interpolated_direct_scatter_proj_data_info_sptr);	
	interpolate_projdata(interpolated_direct_scatter, *scatter_proj_data_sptr, BSpline::linear, remove_interleaving);

	if (est_scale_factor_per_sino == 1)
	  {
	    ProjDataInMemory interpolated_scatter(emission_proj_data_info_sptr);
	    inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);
	    
	    std::cout << "Finding scale factors by sinogram" << std::endl;
	    Array<2,float> scale_factors =
	    scale_factors_per_sinogram(
				       *emission_proj_data_sptr, 
				       interpolated_scatter,
				       *attenuation_correct_factors_sptr,
				       attenuation_threshold,
				       mask_radius_in_mm);
	    std::cout << scale_factors;
	    std::cout << "applying scale factors" << std::endl;
	    scale_scatter_per_sinogram(scaled_scatter_proj_data, 
				       interpolated_scatter,
				       scale_factors) ;
	  }
	else if (est_scale_factor_per_sino == 2)
	  {
	    ProjDataInMemory interpolated_scatter(emission_proj_data_info_sptr);
	    inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);
	    
	    std::cout << "Finding scale factors by viewgram" << std::endl;
	    Array<2,float> scale_factors =
	    scale_factors_per_viewgram(
				       *emission_proj_data_sptr, 
				       interpolated_scatter,
				       *attenuation_correct_factors_sptr,
				       attenuation_threshold,
                                       mask_radius_in_mm);
	    std::cout << scale_factors;
	    std::cout << "applying scale factors" << std::endl;
	    scale_scatter_per_viewgram(scaled_scatter_proj_data, 
				       interpolated_scatter,
				       scale_factors) ;
	  }
	else
	  {
	    inverse_SSRB(scaled_scatter_proj_data, interpolated_direct_scatter);
	  }
	return EXIT_SUCCESS;
}                 
