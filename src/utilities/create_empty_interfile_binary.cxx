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

  This utility will not overwrite existing binary files.

  Currently, requires header file to have a valid extension (.hs or .hv) to determine the type of data.

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
#include "stir/ProjData.h"
#include "stir/ProjDataInterfile.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/Succeeded.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/write_to_file.h"

using namespace stir;

Succeeded
create_empty_projdata_interfile_binary(char* header_file, float fill_value)
{
  InterfilePDFSHeader hdr;

  if (!hdr.parse(header_file))
    return Succeeded::no;

  if (std::ifstream(hdr.data_file_name).good())
    {
      warning("create_empty_projdata_interfile_binary: data file already exists");
      return Succeeded::no;
    }
  shared_ptr<ProjData> proj_data_sptr(new ProjDataInterfile(hdr.get_exam_info_sptr(), hdr.data_info_sptr, hdr.data_file_name));
  proj_data_sptr->fill(fill_value);
  return Succeeded::yes;
}

Succeeded
create_empty_discretised_density_interfile_binary(char* header_file, float fill_value)
{
  InterfileImageHeader hdr;
  if (!hdr.parse(header_file))
    return Succeeded::no;

  if (std::ifstream(hdr.data_file_name).good())
    {
      warning("create_empty_discretised_density_interfile_binary: data file already exists");
      return Succeeded::no;
    }

  VoxelsOnCartesianGrid<float> density = create_image_from_header(hdr);
  density.fill(fill_value);
  write_to_file(hdr.data_file_name, density);
  return Succeeded::yes;
}

Succeeded
process_header_file(char* header_file, float fill_value)
{
  // Verify that the header file is valid and can be parsed
  if (!std::ifstream(std::string(header_file).c_str()).is_open())
    error("read_interfile_image: couldn't open file %s\n", header_file);

  if (strlen(header_file) < 3)
    {
      std::cerr << "Warning: Header file name is too short.\n";
      return Succeeded::no;
    }

  std::string header_file_ext = std::string(header_file).substr(strlen(header_file) - 3);
  if (header_file_ext == ".hs")
    {
      std::cerr << "File extension indicates a ProjData. Creating empty ProjData with uniform value: " << fill_value << '\n';
      return create_empty_projdata_interfile_binary(header_file, fill_value);
    }
  else if (header_file_ext == ".hv")
    {
      std::cerr << "File extension indicates a DiscretisedDensity. Creating empty DiscretisedDensity with uniform value: "
                << fill_value << '\n';
      return create_empty_discretised_density_interfile_binary(header_file, fill_value);
    }
  else
    {
      std::cerr << "Error: Invalid header file extension: " << header_file_ext << '\n';
      return Succeeded::no;
    }
}

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

  std::cerr << "Valid configuration:\n\tHeader File:\t" << argv[1] << "\n\tFill Value:\t\t" << fill_value << '\n';

  if (process_header_file(argv[1], fill_value) == Succeeded::no)
    {
      std::cerr << "Error: Failed to create empty interfile binary file.\n";
      return EXIT_FAILURE;
    }
  std::cerr << "Successfully created empty interfile binary file:\n\tHeader File:\t" << argv[1] << "\n\tFill Value:\t\t"
            << fill_value << '\n';

  return EXIT_SUCCESS;
}