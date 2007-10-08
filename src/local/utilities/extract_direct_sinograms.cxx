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
\brief Create 2D projdata from 3D if span>1

\author Kris Thielemans

$Date$
$Revision$

\par Usage:
\code
extract_direct_sinograms out_projdata_filename in_projdata

\endcode	  
*/
#include <iostream>
#include <string>
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Sinogram.h"
USING_NAMESPACE_STIR

int main(int argc, const char *argv[])                                  
{         
  
  if (argc!=3)
    {
      std::cerr << "Usage:" << argv[0] << "\n"		
		<< "\t[out_projdata_filename]\n" 
		<< "\t[in_projdata]\n";
      
      return EXIT_FAILURE;            
    }      		
  const shared_ptr<ProjData> in_proj_data_sptr = ProjData::read_from_file(argv[2]);  
  if (is_null_ptr(in_proj_data_sptr))
    error("Check the input file");		
  shared_ptr<ProjDataInfo> proj_data_info_sptr =
    in_proj_data_sptr->get_proj_data_info_ptr()->clone();
  proj_data_info_sptr->reduce_segment_range(0,0);
  ProjDataInfoCylindrical& proj_data_info =
    dynamic_cast<ProjDataInfoCylindrical&>(*proj_data_info_sptr);

  if (proj_data_info.get_max_ring_difference(0) == 0 && 
      proj_data_info.get_min_ring_difference(0) == 0)
    error("Input data is already span=1");

  if (proj_data_info.get_max_ring_difference(0) != 1 || 
      proj_data_info.get_min_ring_difference(0) != -1)
    warning("Extracted sinograms are not really span=1, but we'll pretend they are anyway");

  proj_data_info.set_min_axial_pos_num(0,0);
  const int new_num_axial_poss= (in_proj_data_sptr->get_num_axial_poss(0)+1)/2;
  proj_data_info.set_max_axial_pos_num(new_num_axial_poss - 1, 0);
  proj_data_info.set_min_ring_difference(0,0);
  proj_data_info.set_max_ring_difference(0,0);

  //std::cerr << proj_data_info.parameter_info();
  
  const std::string out_proj_data_filename(argv[1]);
  ProjDataInterfile out_proj_data(proj_data_info_sptr, out_proj_data_filename);

  for( int out_ax_pos_num=out_proj_data.get_min_axial_pos_num(0),
	 in_ax_pos_num=in_proj_data_sptr->get_min_axial_pos_num(0);
       out_ax_pos_num<=out_proj_data.get_max_axial_pos_num(0);
       ++out_ax_pos_num, in_ax_pos_num+=2)
    {
      
      Sinogram<float> sino = out_proj_data.get_empty_sinogram(out_ax_pos_num,0);
      sino += in_proj_data_sptr->get_sinogram(in_ax_pos_num,0);
      if (out_proj_data.set_sinogram(sino)== Succeeded::no)
	error("Problem writing sinogram %d", out_ax_pos_num);
    }

  return EXIT_SUCCESS;
} 
               
