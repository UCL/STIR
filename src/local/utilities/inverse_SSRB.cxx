//
// $Id$
//
/*
  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*!
\file
\ingroup utilities
\brief Create 4D projdata from 3D using inverse_SSRB

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	   correct_for_scatter [4D_projdata_filename]
						   [3D_projdata]
						   [4D_template]	
	  Output: 4D Projdata .hs .s files with name proj_data_4D	  

	
	  This is a utility program which uses the inverse_SSRB function 
	  in order to create a 4D set of projection data. 
	  See the inverse_SSRB.cxx for more details.

	  \endcode	  
*/

#include <iostream>
#include <iomanip>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/utilities.h"
#include "local/stir/inverse_SSRB.h"
#include "stir/Succeeded.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
#endif
USING_NAMESPACE_STIR
using namespace std;
/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         

	if (argc< 2 || argc>3)
	{
	   cerr << "Usage:" << argv[0] << "\n"		
		   << "\t[projdata_4D_filename]\n" 
		   << "\t[projdata_3D]\n"
		   << "\t[projdata_4D_template]\n" ;

		return EXIT_FAILURE;            
	}      		
	shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);  
	const ProjDataInfo* proj_data_info_ptr =
		dynamic_cast<ProjDataInfo const *>(
		template_proj_data_sptr->get_proj_data_info_ptr());
		
	string proj_data_4D_filename(argv[1]);
	ProjDataInterfile proj_data_4D(proj_data_info_ptr->clone(), proj_data_4D_filename,ios::out);

	const shared_ptr<ProjData> proj_data_3D_sptr = ProjData::read_from_file(argv[2],ios::in);  

	if (proj_data_info_ptr==0 || proj_data_3D_sptr==0)
		error("Check the input files\n");		 

	inverse_SSRB(proj_data_4D, *proj_data_3D_sptr);

	return EXIT_SUCCESS;
}                 
