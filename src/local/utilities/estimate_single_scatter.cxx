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
   find_fwhm_in_image filename [num_maxima][level][dimension][nema]
   \endcode
   \param num_maxima defaults to 1
     
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/shared_ptr.h" // nedded?
#include "stir/DiscretisedDensity.h" // nedded?
#include "stir/DiscretisedDensityOnCartesianGrid.h" // nedded?
#include "local/stir/scatter.h"
#include "stir/ProjDataInfo.h"
#include <iostream>
#include <fstream>
#include <iomanip>  // nedded?
#include <list> // nedded?
#include <algorithm>  // nedded?
#include <string> 
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
using std::setw;
#endif

/***********************************************************/     
int main(int argc, char *argv[])                                  
{         
  USING_NAMESPACE_STIR
  using namespace std;                                                                
  if (argc< 2 || argc>6)
  {
    cerr << "Usage:" << argv[0] << " input_image transmission_image \
		 //   output_proj_data_filename 
			template_proj_data_filename [attenuation_threshold]\
			[maximum_scatter_points][maximum_LoRs]\n"
	     << "\tattenuation_threshold defaults to 1000\n" 
         << "\tmaximum_scatter_points defaults to 100\n"  
         << "\tmaximum_LoRs defaults to 1000\n"  
         << "returns a sinogram file with the single scatter contribution\n\n";		
    return EXIT_FAILURE;            
  } 
  const float attenuation_threshold = argc>=5 ? atoi(argv[4]) : 1000 ;  
  int maximum_scatter_points = argc>=6 ? atoi(argv[5]) : 1000 ;  
  int maximum_LoRs = argc>=7 ? atoi(argv[6]) : 1000 ;  
  
  shared_ptr< DiscretisedDensity<3,float> >  activity_image_sptr= 
  DiscretisedDensity<3,float>::read_from_file(argv[1]);
  DiscretisedDensity<3,float>& activity_image = *activity_image_sptr;  
  
  shared_ptr< DiscretisedDensity<3,float> >  density_image_sptr= 
  DiscretisedDensity<3,float>::read_from_file(argv[2]);
  DiscretisedDensity<3,float>& density_image = *density_image_sptr;  
  
  shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);  
  const ProjDataInfoCylindricalNoArcCorr * proj_data_info_sptr =
	  dynamic_cast<ProjDataInfoCylindricalNoArcCorr const *>(
	    proj_data_sptr->get_proj_data_info_ptr());

  if (proj_data_info_sptr==0 || density_image_sptr==0 || activity_image_sptr==0)
	  error("Check the input files\n");

  ProjDataInterfile output_proj_data(proj_data_info_sptr->clone(),
		                             output_proj_data_filename);
  scatter_viewgram(output_proj_data,
	  activity_image, density_image,
	  max_scatt_points,attenuation_threshold);  


/*

  if (dimension!=0)    
  {
    string output_string;
    string input_string(argv[1]);
    string slices_string(argv[2]);             
    string:: iterator string_iter;
    for(string_iter=input_string.begin(); 
        string_iter!=input_string.end() && *string_iter!='.' ;
        ++string_iter)  
    output_string.push_back(*string_iter);     
    if (argc>=4)
    {
       string level_string(argv[3]);
       output_string +=  '_' + slices_string + "_slices_FW" + level_string + 'M' ;
    }
    else
    output_string +=  '_' + slices_string + "_slices_FWHM" ;  
      
    ofstream out(output_string.c_str()); //output file //
    if(!out)
    {
      cout << "Cannot open text file.\n" ;
      return EXIT_FAILURE;
    }       
    out << "Slice\t Z\tY\t X\tResZ(mm) ResY(mm) ResX(mm) Value\n"; 
    for (short counter=0 ; counter!=num_maxima ; ++counter)
    { 
      out << setw(3) << counter+1 << "\t" 
          << setw(3) << current_iter->voxel_location[1] << "\t"
          << setw(3) << current_iter->voxel_location[2] << "\t"
          << setw(3) << current_iter->voxel_location[3] << "\t" 
 	      << setw(6) << current_iter->resolution[1]<< "\t"
       	  << setw(6) << current_iter->resolution[2]<< "\t"
         << setw(6) << current_iter->resolution[3]<< "\t"
          << setw(9) << current_iter->voxel_value << "\n" ;
      ++current_iter;         
    }
    out.close();       
  } 
  else 
    for (short counter=0 ; counter!=num_maxima ; ++counter)
    {
      cout << counter+1 << ". max: " << setw(6)	<<  current_iter->voxel_value	  
	         << " at: "  << setw(3) << current_iter->voxel_location[1]
		       << " (Z) ," << setw(3) << current_iter->voxel_location[2]
           << " (Y) ," << setw(3) << current_iter->voxel_location[3] << " (X) \n" ;
  	  cout << "  \n The resolution in z axis is "
           << setw(6) << (current_iter->resolution[1])
       	   << ", \n The resolution in y axis is "
     	     << setw(6) << (current_iter->resolution[2])
           << ", \n The resolution in x axis is "
     	     << setw(6) << (current_iter->resolution[3])
           << ", in mm relative to origin. \n \n";  
      ++current_iter;         
    }                
	*/
  return EXIT_SUCCESS;
}                 
