//
//
/*
  Copyright (C) 2016, UCL
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
  \ingroup scatter
  \brief Simulates a coarse scatter sinogram

  \author Nikos Efthimiou

  \par Usage:
  \code
  simulate_scatter parfile
  \endcode
  See stir::ScatterSimulation documentation for the format
  of the parameter file.
*/

#include "stir/scatter/ScatterSimulation.h"
#include "stir/Succeeded.h"
/***********************************************************/

int main(int argc, const char *argv[])
{
  stir::ScatterSimulation scatter_simulation;

  if (argc==2)
    {
      if (scatter_simulation.parse(argv[1]) == false)
        return EXIT_FAILURE;
    }
  else
    scatter_simulation.ask_parameters();

  return scatter_simulation.process_data() == stir::Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;


}

