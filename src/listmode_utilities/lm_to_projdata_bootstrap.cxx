//
// $Id$
//
/*!
  \file 
  \ingroup listmode

  \brief Program to bin listmode data to 3d sinograms using bootstrap
 
  \author Kris Thielemans
  
  $Date$
  $Revision $
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/listmode/LmToProjDataBootstrap.h"

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

