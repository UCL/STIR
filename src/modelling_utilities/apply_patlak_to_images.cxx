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
  \brief Apply the Patlak linear fit using Dynamic Images
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \par Usage:
  \code 
  apply_patlak_to_images output_parametric_image input_dynamic_image [par_file] 
  \endcode
  
  \par
  - The dynamic images will be calibrated only if the calibration factor is given. 
  - The \a if_total_cnt is set to true the Dynamic Image will have the total number of 
    counts while if set to false it will have the \a total_number_of_counts/get_duration(frame_num).
  - The dynamic images will always be in decaying counts.
  - The plasma data is assumed to be in decaying counts.
  
  \sa PatlakPlot.h for the \a par_file

  \note This implementation does not use wighted least squares because for Patlak Plot only the last frames are used, which they usually have the same duration and similar number of counts.

  \todo Reimplement the method for image-based input function.

  \todo Add to the Doxygen documentation a reference to their paper and how exactly this utility works.
*/

#include "stir/CPUTimer.h"
#include "stir/modelling/PlasmaData.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/PatlakPlot.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include <string>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{ 
USING_NAMESPACE_STIR
  PatlakPlot indirect_patlak;

  if (argc==4)
    {
      if (indirect_patlak.parse(argv[3]) == false)
	return EXIT_FAILURE;
    }
  if (argc!=3 && argc!=4)
    {
      std::cerr << "Usage:" << argv[0] << " output_parametric_image input_dynamic_image [par_file] \n";
      return EXIT_FAILURE;
    }
  if (argc==3)
    indirect_patlak.ask_parameters();
  CPUTimer timer;
  timer.start();
  if (indirect_patlak.set_up()==Succeeded::no)
    return EXIT_FAILURE ;
  else
    {  
      shared_ptr<DynamicDiscretisedDensity> 
	dyn_image_sptr(read_from_file<DynamicDiscretisedDensity>(argv[2]));
      const DynamicDiscretisedDensity & dyn_image= *dyn_image_sptr;
#if 1
      shared_ptr<ParametricVoxelsOnCartesianGrid> 
	par_image_sptr(ParametricVoxelsOnCartesianGrid::read_from_file(argv[1]));
      ParametricVoxelsOnCartesianGrid par_image = *par_image_sptr;
#else
      ParametricVoxelsOnCartesianGrid par_image(dyn_image[1]);
#endif
      //ToDo: Assertion for the dyn-par images, sizes I have to create from one to the other image, so then it should be OK...      
      assert(indirect_patlak.get_time_frame_definitions().get_num_frames()==dyn_image.get_time_frame_definitions().get_num_frames());
      indirect_patlak.apply_linear_regression(par_image,dyn_image);

      // Writing image
      std::cerr << "Writing parametric-image in '"<< argv[1] << "'\n";
      const Succeeded writing_succeeded=OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->  
	write_to_file(argv[1], par_image); 
      std::cerr << "Total time for Image-Based Patlak in sec: " << timer.value() <<"\n";
      timer.stop();  
      
      if(writing_succeeded==Succeeded::yes)
	return EXIT_SUCCESS ;
      else 
	return EXIT_FAILURE ;
    }
}

