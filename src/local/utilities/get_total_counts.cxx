//
// $Id$
//
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

/*!
  \file
  \ingroup utilities
  \brief Calculates the total number of counts in a single frame projection_data. 
  \author Charalampos Tsoumpas  

  $Date$
  $Revision$

  \par Usage:
  \code 
  get_total_counts projdata_filename
  \endcode
  \retval total_counts are printed as a nymber of counts over all the segments.

  \todo Estimate total counts in images, as well.
*/

#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"

#include <fstream> 
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ofstream;
using std::cout;
#endif

USING_NAMESPACE_STIR

int 
main(int argc, char **argv)
{
  if(argc!=2)
  {
    std::cerr<< "Usage: " << argv[0] << " projdata \n";
    exit(EXIT_FAILURE);
  }
  shared_ptr<ProjData> proj_data = 0;
  proj_data =  ProjData::read_from_file(argv[1]);   
  const ProjDataInfo * proj_data_info_ptr = 
    (*proj_data).get_proj_data_info_ptr();  
  int min_segment_num = proj_data_info_ptr->get_min_segment_num();
  int max_segment_num = proj_data_info_ptr->get_max_segment_num();     
  double sum=0.;
  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
  segment_num++) 
    for ( int view_num = proj_data_info_ptr->get_min_view_num();
    view_num<=proj_data_info_ptr->get_max_view_num(); view_num++)
      {
	Viewgram<float> viewgram = proj_data->get_viewgram(view_num,segment_num);
	for (int i= viewgram.get_min_axial_pos_num(); i<=viewgram.get_max_axial_pos_num();i++)
	  for (int j= viewgram.get_min_tangential_pos_num() ; j<=viewgram.get_max_tangential_pos_num();j++)
	    sum+=viewgram[i][j];
      }
      std::cout << sum << "\n";

   return EXIT_SUCCESS;
}
