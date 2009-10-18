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
  \author Kris Thielemans
  
  $Date$
  $Revision$
	
  \par Usage:
  \verbatim
  interpolate_projdata projdata_output_filename projdata_input projdata_template \
     [spline_type1 [spline_type2 [spline_type3 [remove_interleaving [use_view_offset]]]]
  Output: [projdata_output_filename].hs [projdata_output_filename].s files
  \endverbatim  
  Execute without arguments to get more detailed usage info.

  \par
  This is a utility program which uses the stir::interpolate_projdata function in order to 
  interpolate a 3D set of projection data using B-Splines interpolation. 
  \sa "local/stir/interpolate_projdata.h" for more details.
  
*/
#include <iostream>
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/BasicCoordinate.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/numerics/BSplines.h"
#include "local/stir/interpolate_projdata.h"

using namespace stir;
using namespace stir::BSpline;
/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         
	
  if (argc< 4 || argc>9)
    {
      std::cerr << "Usage:" << argv[0] << "\\\n"		
		<< "\tprojdata_out_filename\\\n" 
		<< "\tprojdata_in\\\n"
		<< "\tprojdata_out_template\\\n" 
		<< "\t[spline_type1 [spline_type2 [spline_type3 [remove_interleaving [ use_view_offset]]]]\n"
		<< " Default for spline_type1 is cubic splines.\n"
		<< "      For Nearest Neighbour set to: " << BSpline::near_n << "\n"
		<< "      For Linear set to: " << BSpline::linear << "\n"
		<< "      For Quadratic set to: " << BSpline::quadratic << "\n"
		<< "      For Cubic set to: " << BSpline::cubic << "\n"
		<< "      For Cubic oMoms set to: " << BSpline::oMoms << "\n"
		<< " Default for spline_type2 and spline_type3 is spline_type1.\n"
		<< " Default for remove_interleaving is 0.\n"
		<< " Default for use_view_offset is 0 (WARNING: enabling this is currently EXPERIMENTAL).\n";
      return EXIT_FAILURE;            
    }      		
	
  BasicCoordinate<3, BSplineType> these_types ;
  BasicCoordinate<3, int> input_types ;

  input_types[1]= (argc==4) ? 3 : atoi(argv[4]) ;
  input_types[2]= (argc>5 ? atoi(argv[5]) : input_types[1]);	
  input_types[3]= (argc>6 ? atoi(argv[6]) : input_types[1]);	

  const bool remove_interleaving = argc>7 ? atoi(argv[7]) : 0 ; 
  const bool use_view_offset = argc>8 ? atoi(argv[8]) : 0 ; 
  these_types[1]= (BSplineType)input_types[1];
  these_types[2]= (BSplineType)input_types[2];
  these_types[3]= (BSplineType)input_types[3];

  shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(argv[3]);  
  const ProjDataInfo* proj_data_info_ptr =
    template_proj_data_sptr->get_proj_data_info_ptr();
	
  string proj_data_out_filename(argv[1]);
  ProjDataInterfile proj_data_out(proj_data_info_ptr->clone(), proj_data_out_filename,std::ios::out);
	
  const shared_ptr<ProjData> proj_data_in_sptr = ProjData::read_from_file(argv[2],std::ios::in);  
	
  if (proj_data_info_ptr==0 || proj_data_in_sptr==0)
    error("Check the input files\n");		 
  if (interpolate_projdata(proj_data_out, *proj_data_in_sptr, these_types, remove_interleaving, use_view_offset) ==
      Succeeded::yes)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}                 
