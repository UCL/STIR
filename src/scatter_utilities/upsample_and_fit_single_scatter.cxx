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
  \code
  correct_for_scatter [attenuation_image]
  [no_scatter_viewgram]
  [scatter_viewgram]
  [scaled_scatter_filename]
  [attenuation_threshold]
  [global_scale_factor]
							
  Output: Viewgram with name scaled_scatter_filename
              
	  \endcode
	  \param attenuation_threshold defaults to .05 cm^-1	  
*/

#include <iostream>
#include <fstream>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "local/stir/inverse_SSRB.h"
#include "local/stir/Scatter.h"
#include "local/stir/interpolate_projdata.h"
#include "stir/utilities.h"
#include "stir/IndexRange2D.h" 
#include "stir/stream.h"
#include "stir/Succeeded.h"
//#include "stir/Bin.h" 
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
			<< "\t[output_filename]\n"
			<< "\t[attenuation_threshold]\n"
			<< "\t[scale_factor_per_sinogram]\n"
			<< "\tattenuation_threshold defaults to .01 cm^-1\n" 
			<< "\tusing defaults to 1 for scaling per sinogram"	;		
		return EXIT_FAILURE;            
	}      
	const float attenuation_threshold = argc>=6 ? atof(argv[5]) : .01 ;
	const int est_scale_factor_per_sino = argc>=7 ? atoi(argv[6]) : 1 ; 
	
	shared_ptr< DiscretisedDensity<3,float> >  	
		density_image_sptr= 
		DiscretisedDensity<3,float>::read_from_file(argv[1]);
	
	if (density_image_sptr==0)
		error("Check the input files\n");

	warning("\nWARNING: Attenuation image data are supposed to be in units cm^-1\n"
		"\tReference: water has mu .096 cm^-1\n" 
		"\tMax in attenuation image: %g\n" ,
		density_image_sptr->find_max());

	const shared_ptr<ProjData> emission_proj_data_sptr = ProjData::read_from_file(argv[2]);  

	const shared_ptr<ProjData> scatter_proj_data_sptr = ProjData::read_from_file(argv[3]);   
	
	shared_ptr<ProjDataInfo> emission_proj_data_info_sptr =
		emission_proj_data_sptr->get_proj_data_info_ptr()->clone();
	
	string scaled_scatter_filename(argv[4]);    			
	ProjDataInterfile scaled_scatter_proj_data(emission_proj_data_info_sptr, scaled_scatter_filename);
	
	string att_proj_data_filename("att_proj_data");
	ProjDataInterfile att_proj_data(emission_proj_data_info_sptr, att_proj_data_filename,ios::out);

	// interpolate scatter sinogram
	// first call interpolate_projdata to 'expand' segment 0 to appropriate size (i.e. same as emission data)
	// then call inverse_SSRB to generate oblique segments

	shared_ptr<ProjDataInfo> interpolated_direct_scatter_proj_data_info_sptr =
	  emission_proj_data_sptr->get_proj_data_info_ptr()->clone();
	interpolated_direct_scatter_proj_data_info_sptr->reduce_segment_range(0,0);

	ProjDataInMemory interpolated_direct_scatter(interpolated_direct_scatter_proj_data_info_sptr);	
	interpolate_projdata(interpolated_direct_scatter, *scatter_proj_data_sptr, BSpline::linear);

	ProjDataInMemory interpolated_scatter(emission_proj_data_info_sptr);
	inverse_SSRB(interpolated_scatter, interpolated_direct_scatter);

	const DiscretisedDensityOnCartesianGrid<3,float>& density_image = 
		dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float>&  > 
		(*density_image_sptr.get());
	estimate_att_viewgram(att_proj_data, density_image);
	Array<2,float> scale_factors;
	if (est_scale_factor_per_sino==1)
	  scale_factors =
	    scale_factors_per_sinogram(
				       *emission_proj_data_sptr, 
				       interpolated_scatter,
				       att_proj_data,
				       attenuation_threshold);

	std::cerr << scale_factors;
	scale_scatter_per_sinogram(scaled_scatter_proj_data, 
				   interpolated_scatter,
				   scale_factors) ;

	return EXIT_SUCCESS;
}                 
