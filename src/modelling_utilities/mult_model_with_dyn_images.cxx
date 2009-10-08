//
// $Id$
//
/*
  Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \brief Multiplies Dynamic Images with the Model Matrix creating image in the Parametric Space
  \author Charalampos Tsoumpas

  \par Usage:
  \code 
  mult_model_with_dyn_images output_parametric_image input_dynamic_image [par_file] 
  \endcode
  
  \par
  - The dynamic images will be calibrated only if the calibration factor is given. 
  - The dynamic images and the plasma data must be both either in decaying counts or in decay-corrected counts.
  
  \sa PatlakPlot.h for the \a par_file

  \todo Add to the Doxygen documentation how exactly this utility works.

  $Date$
  $Revision$
*/

#include "stir/modelling/PlasmaData.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/PatlakPlot.h"
#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include <string>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR

// Impelemented only for the linear Patlak Plot so far. 
// In the future I should implement  the KineticModels with the "linear" specification 
// for patlak, logan etc...
  PatlakPlot patlak_plot;

  if (argc==4)
    {
      if (patlak_plot.parse(argv[3]) == false)
	{
	  std::cerr << "Usage:" << argv[0] << " output_parametric_image input_dynamic_image [par_file] \n";
	  return EXIT_FAILURE;
	}
    }
  if (argc!=3 && argc!=4)
    return EXIT_FAILURE;
  if (argc==3)
    patlak_plot.ask_parameters();
  if (patlak_plot.set_up()==Succeeded::no)
    return EXIT_FAILURE ;
  else
    {  
      shared_ptr<ParametricVoxelsOnCartesianGrid> par_image_sptr =
	ParametricVoxelsOnCartesianGrid::read_from_file(argv[1]);
      ParametricVoxelsOnCartesianGrid par_image = *par_image_sptr;

      shared_ptr<DynamicDiscretisedDensity> dyn_image_sptr =
	DynamicDiscretisedDensity::read_from_file(argv[2]);
      const DynamicDiscretisedDensity & dyn_image= *dyn_image_sptr;

      //NotToDo: Assertion for the dyn-par images, sizes should not be ncessary ONLY WHEN I will create from dyn_image the par_image...
      assert(patlak_plot.get_time_frame_definitions().get_num_frames()==dyn_image.get_time_frame_definitions().get_num_frames());
      patlak_plot.multiply_dynamic_image_with_model_gradient(par_image,dyn_image);

  // Writing image 
      std::cerr << "Writing parametric-image in '"<< argv[1] << "'\n";
      const Succeeded writing_succeeded=OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->  
	write_to_file(argv[1], par_image); 

   if(writing_succeeded==Succeeded::yes)
     return EXIT_SUCCESS ;
   else 
     return EXIT_FAILURE ;
    }
}

