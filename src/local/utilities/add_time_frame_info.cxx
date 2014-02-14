
//
// $id: apply_patlak_to_images.cxx,v 1.1 2005/12/02 16:22:23 ctsoumpas Exp $
//
/*!
  \file
  \ingroup utilities
  \brief Add the Time Frame Information to Dynamic Images
  \author Charalampos Tsoumpas
*/
/*
  Copyright (C) 2006- 2007, Hammersmith Imanet Ltd
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

#include "stir/CPUTimer.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/TimeFrameDefinitions.h"
#include "local/stir/DynamicDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::ifstream;
using std::istream;
using std::setw;
#endif

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
    if (argc!=3)
      {
	std::cerr << "Usage:" << argv[0] << "\n"
		  << "\t[dynamic_image_filename]\n"
		  << "\t[time_frames_filename]\n\n";
	return EXIT_FAILURE;            
      }       

  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
    DynamicDiscretisedDensity::read_from_file(argv[1]);
  DynamicDiscretisedDensity dyn_image = *dyn_image_sptr;
  const TimeFrameDefinitions frame_defs(argv[2]);
  assert(frame_defs.get_num_frames()==dyn_image.get_num_time_frames());
  dyn_image.set_time_frame_definitions(frame_defs);
  const TimeFrameDefinitions time_defs=dyn_image.get_time_frame_definitions();
  assert(frame_defs.get_duration(1)==time_defs.get_duration(1));
  std::cerr << "Duration Time " << dyn_image.get_time_frame_definitions().get_duration(2) << "\n";
  Succeeded writing_succeeded=dyn_image.write_to_ecat7(argv[1]);
  if(writing_succeeded==Succeeded::yes)
    return EXIT_SUCCESS ;
  else 
    return EXIT_FAILURE ;
}
