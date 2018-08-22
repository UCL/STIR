//
//
/*
  Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
  Copyright (C) 2018, University College London
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
  \brief Multiplies Parametric Images with the Model Matrix creating Dynamic Images
  \author Charalampos Tsoumpas  
  \author Richard Brown

  \par Usage:
  \code 
  get_dynamic_images_from_parametric_images output_parametric_image input_dynamic_image [par_file] 
  \endcode
  
  \par
  - The dynamic images will be calibrated only if the calibration factor is given. 
  - The dynamic images will be in decaying counts if the plasma data are in decaying counts.
  
  \sa PatlakPlot.h for the \a par_file

  \todo Add to the Doxygen documentation how exactly this utility works.

*/

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/PatlakPlot.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR

// Impelemented only for the linear Patlak Plot so far. 
// In the future I should implement  the KineticModels with the "linear" specification 
// for patlak, logan etc...  PatlakPlot patlak_plot;
  PatlakPlot patlak_plot;

  if (argc==4)
    {
      if (patlak_plot.parse(argv[3]) == false)
	return EXIT_FAILURE;
    }
  if (argc!=3 && argc!=4)
    {
      std::cerr << "Usage:" << argv[0] << " output_dynamic_image input_parametric_image [par_file] \n";
      return EXIT_FAILURE;
    }
  if (argc==3)
    patlak_plot.ask_parameters();
  if (patlak_plot.set_up()==Succeeded::no)
    return EXIT_FAILURE ;
  else
    {  
      shared_ptr<ParametricVoxelsOnCartesianGrid> 
        par_image_sptr(ParametricVoxelsOnCartesianGrid::read_from_file(argv[2]));

      // Get all the info we need
      const ExamInfo                            exam_info       = par_image_sptr->get_exam_info();
      const TimeFrameDefinitions                tdefs           = patlak_plot.get_time_frame_definitions();
      const double                              time_since_1970 = exam_info.start_time_in_secs_since_1970;
      shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name(par_image_sptr->get_exam_info().originating_system));
      shared_ptr<VoxelsOnCartesianGrid<float> > voxels_sptr(par_image_sptr->construct_single_density(1).clone());

      // Construct the dynamic image
      DynamicDiscretisedDensity dyn_image(tdefs,
                                          time_since_1970,
                                          scanner_sptr,
                                          voxels_sptr);

      patlak_plot.get_dynamic_image_from_parametric_image(dyn_image,*par_image_sptr);

  // Writing image 
  std::cerr << "Writing dynamic-image in '"<< argv[1] << "'\n";

  Succeeded writing_succeeded = OutputFileFormat<DynamicDiscretisedDensity>::default_sptr()->write_to_file(argv[1], dyn_image);

  if(writing_succeeded==Succeeded::yes)
    return EXIT_SUCCESS ;
  else 
    return EXIT_FAILURE ;
    }
}


