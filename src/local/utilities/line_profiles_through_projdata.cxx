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


 /* This program takes the requested profile of which is then written to a file.   */

#include "stir/Array.h"
#include "stir/Viewgram.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <iostream> 
#include <fstream>
#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
#endif
using namespace std;
USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
	if (argc!=2)
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
					
    float view_mean=(view_min+view_max)/2.F;
    float axial_mean=(axial_min+axial_max)/2.F;

    Viewgram<float> profile_viewgram = input_projdata_sptr->get_viewgram(view_mean,0,0);
	
	ofstream profile_stream(output_profile_string.c_str(), ios::out); //output file //
		if(!profile_stream)    
			cerr << "Cannot open " << output_profile_string << ".prof file.\n" ;
		else
			for (int tang=tangential_min ; tang<= tangential_max ; ++tang)		
				profile_stream  << profile_viewgram[axial_mean][tang] << "  ";
		profile_stream.close();   
	return EXIT_SUCCESS;
}

