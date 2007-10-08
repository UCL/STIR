
//
// $id: apply_logan_to_images.cxx,v 1.1 2005/12/02 16:22:23 ctsoumpas Exp $
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
  \brief Apply the Logan linear fit using Dynamic Images
  \author Charalampos Tsoumpas

  $Date$
  $Revision$

  \todo This methods needs re-implementation to take into account the changes like un the \sa apply_patlak_to_images.cxx
*/

#include "stir/CPUTimer.h"
#include "local/stir/modelling/PlasmaData.h"
#include "local/stir/modelling/PlasmaSample.h"
#include "local/stir/modelling/BloodFrame.h"
#include "local/stir/modelling/BloodFrameData.h"
#include "local/stir/modelling/ParametricDiscretisedDensity.h"
#include "local/stir/modelling/logan.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include <string>
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
  
    if (argc<4 || argc>10)
      {
	std::cerr << "Usage:" << argv[0] << "\n"
		  << "\t[parametric_image_filename]\n"
		  << "\t[dynamic_image_filename]\n"
		  << "\t[plasma_data_filename]\n" 
		  << "\t[input function time_shift in sec]\n" 
		  << "\t[blood volume (bv)]\n" 
		  << "\t[Num of frames for Logan]\n" 
		  << "\t[is_decay_corrected]\n"
		  << "\t[is_calibrated]\n"
		  << "\t[nnls]\n"
		  << "\ttime_shift: default to 13 sec\n"
		  << "\tbv: default to 0.05\n" 
		  << "\tlogan_num_frame: default 13 to take the last 13 frames.\n"
		  << "\tis_decay_corrected: is set to false for both PlasmaData and DynamicDiscretisedDensity. \n"
  		  << "\tis_calibrated: is set to true. False will cause an error. \n"
		  << "\tnlls: is set to true for Non-Linear-Least-Squeares fit. \n\t\tFalse is default which runs Ordinary-Least-Squares.\n";
	return EXIT_FAILURE;            
      }       
#if 1
    else 
      return EXIT_SUCCESS;
#else
  CPUTimer timer;
  timer.start();
  const float time_shift = argc>=5 ? atof(argv[4]) : 13.F ;
  const bool nlls = argc>=10 ? atoi(argv[9]) : false ;
  const float bv = argc>=6 ?  atof(argv[5]) : 0.05F ; // Blood Volume, usually constant. ChT::ToDo:Maybe add to the PlasmaData class!!! //ChT::Check: If this works as I expect.
  const unsigned int logan_num_frames = argc>=7 ? atoi(argv[6]) : 13 ;
  const bool is_decay_corrected = argc>=8 ? atoi(argv[7]) : false ;
  const bool is_calibrated = argc>=9 ? atoi(argv[8]) : true ;

  //Read Dynamic Sequence of ECAT7 Images, in respect to their center in x, y axes as origin
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr= 
    DynamicDiscretisedDensity::read_from_file(argv[2]);
  DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;
  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  if (num_frames<=logan_num_frames)
    error("Logan implementation is trying to take frames more than the total frames of the study!\n");
  shared_ptr<DiscretisedDensity<3,float> > slope_image_sptr = 
    dyn_image_sptr->get_density(1).clone();
  shared_ptr<ParametricVoxelsOnCartesianGrid> parametric_image_sptr =
    ParametricVoxelsOnCartesianGrid::read_from_file(argv[1]);
  ParametricVoxelsOnCartesianGrid & parametric_image = *parametric_image_sptr;

  // Read the Plasma Data from the given file.
  // Shift the Plasma Data as appropriately. 
  PlasmaData plasma_data;
      plasma_data.read_plasma_data(argv[3]);   // The implementation assumes three list file of blood. 
      plasma_data.shift_time(time_shift);
#if 0
      apply_logan_to_images(parametric_image, 
			    dyn_image,  
			    plasma_data, 
			    time_shift, 
			    num_frames-logan_num_frames+1, 
			    bv, 
			    is_calibrated, 
			    is_decay_corrected);
#endif
  // Writing image
  std::cerr << "Writing 'parametric-images" << "\n";
  Succeeded writing_succeeded=OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->
    write_to_file(argv[1], *parametric_image_sptr);  
  std::cerr << "Total time for Image-Based Logan in sec: " << timer.value() <<"\n";
  timer.stop();  
   if(writing_succeeded==Succeeded::yes)
     return EXIT_SUCCESS ;
   else 
     return EXIT_FAILURE ;
#endif
}

