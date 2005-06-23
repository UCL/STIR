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
\brief Interpolate projdata using BSplines

  \author Charalampos Tsoumpas
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	  interpolate_projdata [projdata_output_filename]
	  [projdata_input]
	  [projdata_template]
	  [spline_type]
	  
		Output: "projdata_output".hs "projdata_output_filename".s files
		
		  This is a utility program which uses the interpolate_projdata function 
		  in order to interpolate a 3D set of projection data using B-SPlines interpolation. 
		  See the ../numerics_buildblock/interpolate_projdata.cxx for more details.
		  \endcode	  
*/
#include <iostream>
#include <iomanip>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/BasicCoordinate.h"
#include "stir/utilities.h"
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "local/stir/interpolate_projdata.h"
#include "stir/Succeeded.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;
#endif

using namespace stir;
using namespace stir::BSpline;
using namespace std;
/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         
	
	if (argc< 3 || argc>7)
	{
		cerr << "Usage:" << argv[0] << "\n"		
			<< "\t[projdata_out_filename]\n" 
			<< "\t[projdata_in]\n"
			<< "\t[projdata_out_template]\n" 
			<< "\t[spline_type[1]]\n"
			<< "\t[spline_type[2]]\n"
			<< "\t[spline_type[3]]\n";
		return EXIT_FAILURE;            
	}      		
	
	BasicCoordinate<3, BSplineType> these_types ;
	these_types[1]= cubic;//argc==3 ? 3 : atoi(argv[4]) ;
	these_types[2]= cubic;//argc==3 ? 3 : (=argc>4 ? atoi(argv[5]) : these_types[1]);	
	these_types[3]= cubic;//argc==3 ? 3 : (=argc>4 ? atoi(argv[6]) : these_types[1]);	
	
	shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);  
	const ProjDataInfo* proj_data_info_ptr =
		dynamic_cast<ProjDataInfo const *>(
		template_proj_data_sptr->get_proj_data_info_ptr());
	
	string proj_data_out_filename(argv[1]);
	ProjDataInterfile proj_data_out(proj_data_info_ptr->clone(), proj_data_out_filename,ios::out);
	
	const shared_ptr<ProjData> proj_data_in_sptr = ProjData::read_from_file(argv[2],ios::in);  
	
	if (proj_data_info_ptr==0 || proj_data_in_sptr==0)
		error("Check the input files\n");		 
	interpolate_projdata(proj_data_out, *proj_data_in_sptr, these_types);
	
	return EXIT_SUCCESS;
}                 
