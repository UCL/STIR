/*
  Copyright (C) 2020, University College London
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief create a "Multi" header for a set of data

  This is a command line utility for creating a header that "points" to several other
  data-sets, useful for e.g. dynamic images/data. The output header can then be used in
  other STIR utilities. This is useful for instance if you have
  a set of reconstruct images corresponding to several time frames, but need to read the data
  as a single study.

  The command line arguments are as follows:
  \code
  output_filename_with_extension in_data1 [in_data2 [in_data3...]]
  \endcode

  \warning There is no check that the data sizes and other info are compatible.
  Hence, lots of funny effects can happen if data are not compatible.
  \author Kris Thielemans 
*/

#include "stir/MultipleDataSetHeader.h"
#include <vector>
#include <fstream> 
#include <iostream> 

USING_NAMESPACE_STIR

int 
main(int argc, char **argv)
{
  if(argc<3)
    {
      std::cerr<< "Usage: " << argv[0] << "\n\t"
               << "output_filename_with_extension in_data1 [in_data2 [in_data3...]]\n\n";
      exit(EXIT_FAILURE);
    }
  // skip program name
  --argc;
  ++argv;


  if (argc==0)
    { std::cerr << "No output file (nor input files) on command line\n"; exit(EXIT_FAILURE); }

  // find output filename
  const std::string output_file_name = *argv;
    {
      --argc; ++argv;
    }

  const int num_files = argc;
  if (num_files==0)
    { std::cerr << "No input files on command line\n"; exit(EXIT_FAILURE); }

  MultipleDataSetHeader::write_header(output_file_name, std::vector<char*>(argv, argv+argc));

  return EXIT_SUCCESS;
}
