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
  if (argc!=9)
    {
      cerr << "Usage:" << argv[0] << "\n"
	   << "\t[proj_data_filename]\n" 
	   << "\t[output_profile_filename] ax_min/max view_min/max tang_min/max\n";
      return EXIT_FAILURE;            
    }      
	
  const shared_ptr<ProjData> input_projdata_sptr = ProjData::read_from_file(argv[1]);  
 
  const ProjDataInfo * projdata_info_ptr = 
    (*input_projdata_sptr).get_proj_data_info_ptr();

  string output_profile_string(argv[2]);

  const int axial_min = atoi(argv[3]);
  const int axial_max = atoi(argv[4]);
  const int view_min = atoi(argv[5]);
  const int view_max = atoi(argv[6]);
  const int tangential_min = atoi(argv[7]);
  const int tangential_max = atoi(argv[8]);
  /*
  int view_min, tangential_min, 
    view_max, tangential_max;

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
			
  */
  /*  int view_mean=static_cast<int>(ceil((view_min+view_max)/2.F));
  //int axial_mean=static_cast<int>(floor((axial_min+axial_max)/2.F));
  int tangential_mean=static_cast<int>(floor((tangential_min+tangential_max)/2.F));
  */
  Sinogram<float> profile_sinogram = input_projdata_sptr->get_sinogram(axial_min,0);
  for (int ax = axial_min+1; ax<=axial_max; ++ax)
    profile_sinogram += input_projdata_sptr->get_sinogram(ax,0);
  profile_sinogram /= (axial_max-axial_min+1);

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
	Array<1,float> prof = profile_sinogram[view_min];
	for (int view_num=view_min+1 ; view_num<= view_max ; ++view_num)
	  prof +=profile_sinogram[view_num];
	prof /= (view_max-view_min+1);
	for (int tang=projdata_info_ptr->get_min_tangential_pos_num() ; 
	     tang<=projdata_info_ptr->get_max_tangential_pos_num() ; ++tang)
	  {
	    Bin bin(0,view_min,axial_min,tang,0);
	    profile_stream  << std::setw(9) << projdata_info_ptr->get_s(bin) << "\t"
			    << std::setw(7) << prof[tang] << endl ;
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
	for (int view_num=projdata_info_ptr->get_min_view_num() ; 
	     view_num<= projdata_info_ptr->get_max_view_num();
	     ++view_num)		
	  {
	    float value=0;
	    for (int tang=tangential_min ; tang<= tangential_max ; ++tang)		
	      value += profile_sinogram[view_num][tang];
	    value /= tangential_max-tangential_min+1;
		  Bin bin(0,view_num,axial_min,tangential_min,0);
	    profile_stream  << std::setw(9) << projdata_info_ptr->get_phi(bin) << "\t"
			    << std::setw(7) << value << endl ;
	  }
	profile_stream.close();   
      }
  }

  return EXIT_SUCCESS;
}

