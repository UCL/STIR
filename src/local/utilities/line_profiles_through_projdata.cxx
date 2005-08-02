//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Extracting profiles from projection data
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/


 /* This program extracts the tangential profile of given input_projdata which should be on the center. 
    The results is the same as if we first run ./extract_segments input_projdata.hs (by sinogram) and then
    run ./manip_image input_projdata.hs to extract the rows as the defaults.   */

#include "stir/Array.h"
#include "stir/Sinogram.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include <iostream> 
#include <iomanip> 
#include <fstream>
#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
using std::setw;
#endif


int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
    using namespace std;
  if (argc!=3)
    {
      cerr << "Usage:" << argv[0] << "\n"
	   << "\t[proj_data_filename]\n" 
	   << "\t[output_profile_filename]\n";
      return EXIT_FAILURE;            
    }      
	
  const shared_ptr<ProjData> input_projdata_sptr = ProjData::read_from_file(argv[1]);  
 
  const ProjDataInfo * projdata_info_ptr = 
    (*input_projdata_sptr).get_proj_data_info_ptr();

  string output_profile_string(argv[2]);

  int view_min, axial_min, tangential_min, 
    view_max, axial_max, tangential_max;

  view_min=
    projdata_info_ptr->get_min_view_num();
  view_max=
    projdata_info_ptr->get_max_view_num();	
  axial_min=
    projdata_info_ptr->get_min_axial_pos_num(0);
  axial_max=
    projdata_info_ptr->get_max_axial_pos_num(0);
  tangential_min=
    projdata_info_ptr->get_min_tangential_pos_num();
  tangential_max=
    projdata_info_ptr->get_max_tangential_pos_num();
			
			
  int view_mean=static_cast<int>(ceil((view_min+view_max)/2.F));
  int axial_mean=static_cast<int>(floor((axial_min+axial_max)/2.F));
  int tangential_mean=static_cast<int>(floor((tangential_min+tangential_max)/2.F));
	
  const Sinogram<float> profile_sinogram = input_projdata_sptr->get_sinogram(axial_mean,0);
	
  // along tangential direction
  {
    const std::string name=output_profile_string+"_tang.prof";
    ofstream profile_stream(name.c_str(), ios::out); //output file //
    if(!profile_stream)    
      cerr << "Cannot open " << name << endl ;
    else
      //	" X-axis"<<
      {
	profile_stream  << " s-Value (mm)" << "\t" <<  "Value" << endl ;

	for (int tang=tangential_min ; tang<= tangential_max ; ++tang)		
	  {
	    Bin bin(0,view_mean,axial_mean,tang,0);
	    profile_stream  << std::setw(9) << projdata_info_ptr->get_s(bin) << "\t"
			    << std::setw(7) << profile_sinogram[view_mean][tang] << endl ;
	  }
	profile_stream.close();   
      }
  }
  // along view direction
  {
    const std::string name=output_profile_string+"_view.prof";
    ofstream profile_stream(name.c_str(), ios::out); //output file //
    if(!profile_stream)    
      cerr << "Cannot open " << name << endl ;
    else
      //	" X-axis"<<
      {
	profile_stream  << " phi (radians)" << "\t" <<  "Value" << endl ;
	      
	for (int view_num=view_min ; view_num<= view_max ; ++view_num)		
	  {
	    Bin bin(0,view_num,axial_mean,tangential_mean,0);
	    profile_stream  << std::setw(9) << projdata_info_ptr->get_phi(bin) << "\t"
			    << std::setw(7) << profile_sinogram[view_num][tangential_mean] << endl ;
	  }
	profile_stream.close();   
      }
  }

  return EXIT_SUCCESS;
}

