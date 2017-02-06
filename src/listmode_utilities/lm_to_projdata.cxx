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

#include "stir/listmode/LmToProjData.h"
#include "stir/IO/InputFileFormatRegistry.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

USING_NAMESPACE_STIR



int main(int argc, char * argv[])
{
  
  if (strcmp(argv[1], "--help") == 0 ||
          strcmp(argv[1], "-?") == 0) {
    cerr <<endl<< "Usage: " << argv[0] << " [par_file]\n"
                                    "Run "<<argv[0]<<" --input-formats to list the supported input formats\n";
    exit(EXIT_FAILURE);
  }
  // N.E: Display the supported inputs, we need this in order to know
  // which listmode files are supported
  if (strcmp(argv[1], "--input-formats")==0)
  {
      cerr<<"Supported input file formats:\n";
      InputFileFormatRegistry<CListModeData>::default_sptr()->
              list_registered_names(cerr);
      exit(EXIT_SUCCESS);
  }

  if (strcmp(argv[1], "--test_timing_positions")==0)
  {
      cerr<<"A test function for TOF data which I could not fit anywhere else right now:\n"
            "It is going to fill every segment with the index number of the respective TOF position \n"
            "and then stop.\n";
       std::cout<<argc << std::endl;
       std::cout << argv[0] << "\n" << argv[1] << "\n" << argv[2] << std::endl;
      LmToProjData application(argc==3 ? argv[2] : 0);
      application.run_tof_test_function();
      exit(EXIT_SUCCESS);
  }

  LmToProjData application(argc==2 ? argv[1] : 0);
  application.process_data();

  return EXIT_SUCCESS;
}

