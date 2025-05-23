//
//
/*!
  \file 
  \ingroup listmode_utilities

  \brief Program to bin listmode data to 3d sinograms

  \see class stir::LmToProjData for info on parameter file format

  \author Kris Thielemans
  \author Sanida Mustafovic
  
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/LmToProjData.h"
#include "stir/IO/InputFileFormatRegistry.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

USING_NAMESPACE_STIR



int main(int argc, char * argv[])
{
  if (argc>1)
    {
      if (strcmp(argv[1], "--help") == 0 ||
          strcmp(argv[1], "-?") == 0) {
	cerr << "\nUsage: " << argv[0] << " [par_file]\n"
	     << "Run "<<argv[0]<<" --input-formats to list the supported input formats\n";
	exit(EXIT_SUCCESS);
      }
      // Display the supported inputs, we need this in order to know
      // which listmode files are supported
      if (strcmp(argv[1], "--input-formats")==0)
	{
	  cerr<<endl<<"Supported input file formats:\n";
      InputFileFormatRegistry<ListModeData>::default_sptr()->
	    list_registered_names(cerr);
	  exit(EXIT_SUCCESS);
	}
    }
  LmToProjData application(argc==2 ? argv[1] : 0);
  application.process_data();

  return EXIT_SUCCESS;
}

