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
  write_patlak_matrix [par_file] 
  \endcode
  \note It writes it always to the text file: "model_matrix.out" of the working directory

  \sa PatlakPlot.h for the \a par_file

  $Date$
  $Revision$

*/

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/PatlakPlot.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
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

  if (argc==2)
    {
      if (patlak_plot.parse(argv[1]) == false)
	return EXIT_FAILURE;
    }
  else
    patlak_plot.ask_parameters();
  if (patlak_plot.set_up()==Succeeded::no)
    {
      std::cerr << "Usage:" << argv[0] << " [par_file] \n";
      return EXIT_FAILURE ;
    }
  else
    {  
  // Writing model matrix  
  std::cerr << "Writing Patlak Model Matrix in file 'model_matrix.out'" << "\n";
  Succeeded writing_succeeded=(patlak_plot.get_model_matrix().write_to_file("model_matrix.out"));

   if(writing_succeeded==Succeeded::yes)
     return EXIT_SUCCESS ;
   else 
     return EXIT_FAILURE ;
    }
}

