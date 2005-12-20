//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Extracting plasma data
  \author Charalampos Tsoumpas
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
/* This program extracts time and activity input plasma data. 
 */
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/modelling/PlasmaSample.h"
#include "local/stir/modelling/BloodFrame.h"
#include "local/stir/modelling/BloodFrameData.h"
#include "stir/Array.h"

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
    if (argc<2 || argc>3)
      {
	std::cerr << "Usage:" << argv[0] << "\n"
		  << "\t[data_filename]\n" 
		  << "\t[default 1 for plasma_data, else 0 for frame_blood_data]\n" ;
	return EXIT_FAILURE;            
      }       

  const bool plasma_type = argc>=3 ? atoi(argv[2]) : true ;

  if(plasma_type)
    {
      PlasmaData testing_plasma_data;
      testing_plasma_data.read_plasma_data(argv[1]);
      testing_plasma_data.shift_time(13.F);
      //DecayCorrect

      PlasmaData::const_iterator cur_iter;
      for (cur_iter=testing_plasma_data.begin(); 
	   cur_iter!=testing_plasma_data.end() ; ++cur_iter)
	std::cout << (*cur_iter).get_time_in_s() << " " << (*cur_iter).get_plasma_counts_in_kBq() <<  " " << (*cur_iter).get_blood_counts_in_kBq() << std::endl ;
    }

  if(!plasma_type)
    {
      BloodFrameData testing_blood_frame_data;
      testing_blood_frame_data.read_blood_frame_data(argv[1]);
      //DecayCorrect
      BloodFrameData::const_iterator cur_iter;
      for (cur_iter=testing_blood_frame_data.begin(); 
	   cur_iter!=testing_blood_frame_data.end() ; ++cur_iter)
	std::cout << (*cur_iter).get_frame_num() << " " << (*cur_iter).get_blood_counts_in_kBq() << std::endl ;
    }

      return EXIT_SUCCESS;
    }

  
