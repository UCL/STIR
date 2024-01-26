/*!
  \file
  \ingroup utilities

  \brief A utility that takes an interfile header and create a corresponding binary file.
  \par Usage
  \verbatim
  create_empty_interfile_binary header_filename [fill_value]
  \endverbatim
  Given a valid Interfile format header, this utility will create a binary file
  with the same size as the data in the header. The optional fill_value argument
  However, the binary file will not contain any data.

  This utility is mainly useful to an empty interfile binary file that can then
  be used for collaborative development. If one user experiences issues with STIR,
  the only the header file needs to be transferred to the other user for debugging issues.

  \author Robert Twyman
*/
/*
    Copyright (C) 2024, Prescient Imaging # TODO: CONFIRM?
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

int
main(int argc, char* argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " header_filename [fill_value=0.0]\n";
      return EXIT_FAILURE;
    }

  std::ifstream header_file(argv[1]);
  if (!header_file.good())
    {
      std::cerr << "Error: Header file does not exist.\n";
      return EXIT_FAILURE;
    }

  float fill_value = 0.0f;
  if (argc == 3)
    {
      try
        {
          fill_value = std::stof(argv[2]);
        }
      catch (const std::invalid_argument& e)
        {
          std::cerr << "Error: Invalid fill_value argument: " << argv[2] << '\n';
          return EXIT_FAILURE;
        }
    }

  std::cerr << "Valid configuration: " << argv[1] << " " << fill_value << '\n';
  

  return EXIT_SUCCESS;
}