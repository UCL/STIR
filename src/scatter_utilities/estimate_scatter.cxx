//
// $Id$
//
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief   

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

#include "local/stir/ScatterEstimationByBin.h"
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

