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
   find_fwhm_in_image filename [num_maxima][level][dimension][nema]
   \endcode
   \param num_maxima defaults to 1
     
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
//#include "stir/Timer.h"
//#include "stir/CPUTimer.h"
#include "local/stir/Scatter.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
//using std::setw;
#endif

/***********************************************************/     

int main(int argc, char *argv[])                                  
{         
  USING_NAMESPACE_STIR
  using namespace std;
  if (argc< 3 || argc>6)
  {
    cerr << "Usage:" << argv[0] << "\n"
     << "\t[activity_image]\n"
		 << "\t[attenuation_image]\n"
		 <<	"\t[proj_data_filename]\n" 
		 << "\t[attenuation_threshold]\n"
		 <<	"\t[maximum_scatter_points]\n"
		 << "\t[maximum_LoRs]\n"
	   << "\t[attenuation_threshold] defaults to .09 cm^-1\n"
     << "\t[maximum_scatter_points] defaults to 1000\n" ;
      // << "\tmaximum_LoRs defaults to 1000\n"
    return EXIT_FAILURE;            
  }      
  float attenuation_threshold = argc>=5 ? atof(argv[4]) : 0.09 ;
  int max_scat_points = argc>=6 ? atoi(argv[5]) : 1000 ,
      maximum_LoRs = argc>=7 ? atoi(argv[6]) : 1000 ;  
    
  shared_ptr< DiscretisedDensity<3,float> >  
	  activity_image_sptr= 
  DiscretisedDensity<3,float>::read_from_file(argv[1]), 
      density_image_sptr= 
  DiscretisedDensity<3,float>::read_from_file(argv[2]);
  
  warning("\nWARNING:\n"
    "\tattenuation image data are supposed to be in units cm^-1\n"
    "\tReference: water has mu .096 cm^-1\n" 
    "\tMax in attenuation image: %g\n" ,
    density_image_sptr->find_max());
#ifndef NORESCALE
  /*
    cerr << "WARNING: multiplying attenuation image by x-voxel size "
    << " to correct for scale factor in forward projectors...\n";
  */
  // projectors work in pixel units, so convert attenuation data 
  // from cm^-1 to pixel_units^-1
  const float rescale = 
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> *>(density_image_sptr.get())->
    get_grid_spacing()[3]/10;
#else
  const float rescale = 
    10.F;
#endif
  *density_image_sptr *= rescale;
  attenuation_threshold *= rescale;
  cout << "\nAtt_thresh = " << attenuation_threshold << "  "<<  rescale << endl;

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
 
    string output_proj_data_filename;
    string input_string(argv[1]);
    replace_extension(input_string,"");
	
	if (argc>=6)
	{
		string max_scat_points_string(argv[5]);             
		output_proj_data_filename =  input_string + '_' +  max_scat_points_string ;	
	}

  ProjDataInterfile output_proj_data(proj_data_info_ptr->clone(),output_proj_data_filename);

  cout << "\nwriting the single scatter contribution into the file: " << output_proj_data_filename <<".s ...\n";		

  scatter_viewgram(output_proj_data,
	  activity_image, density_image,
	  max_scat_points,attenuation_threshold);        
     
  return EXIT_SUCCESS;
}                 
