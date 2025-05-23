//
//
/*!
  \file 
  \ingroup listmode

  \brief Program to bin listmode data to projection data using bootstrapping (uses stir::LmToProjDataBootstrap)
 
  \author Kris Thielemans
  
  $Revision $
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/LmToProjDataBootstrap.h"

USING_NAMESPACE_STIR



int main(int argc, char * argv[])
{
  
  if (argc<1 && argc>3) {
    std::cerr << "Usage: " << argv[0] << " [par_file [seed]]\n";
    exit(EXIT_FAILURE);
  }

  // clumsy way of having extra argument
  if (argc==3)
    {
      LmToProjDataBootstrap<LmToProjData> 
	application(argc>=2 ? argv[1] : 0,
		    atoi(argv[2]));
      application.process_data();
    }
  else
    {
      LmToProjDataBootstrap<LmToProjData> 
	application(argc==2 ? argv[1] : 0);
      application.process_data();
    }
  return EXIT_SUCCESS;
}

