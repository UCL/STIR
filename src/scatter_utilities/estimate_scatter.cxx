//
// $Id$
//
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
  \brief Estimates a coarse scatter sinogram

  \author Kris Thielemans
  
  $Date$
  $Revision$
        
  \par Usage:
  \code
  estimate_scatter parfile
  \endcode
  See stir::ScatterEstimationByBin documentation for the format
  of the parameter file.
*/

#include "stir/scatter/ScatterEstimationByBin.h"
#include "stir/Succeeded.h"
/***********************************************************/     

int main(int argc, const char *argv[])                                  
{         
  stir::ScatterEstimationByBin scatter_estimation;

  if (argc==2)
    {
      if (scatter_estimation.parse(argv[1]) == false)
        return EXIT_FAILURE;
    }
  else
    scatter_estimation.ask_parameters();

  return scatter_estimation.process_data() == stir::Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;


}

