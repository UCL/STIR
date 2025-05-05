//
//
/*
  Copyright (C) 2018, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief Split dynamic image into individual images
  \author Richard Brown

  \par Usage:
  \code
  extract_single_images_from_dynamic_image output_filename_pattern input_header_filename output_format_parameter_file

  The output filename should look something like this: dyn_im_{}_output.file_extension,
  so that we can use boost format. In this fashion, you can can specify the output file extension
  should you wish.

  An example of an output parameter file is as follows:
    OutputFileFormat Parameters:=
    output file format type := interfile
    interfile Output File Format Parameters:=
    number format := float
    number_of_bytes_per_pixel:=4
    End Interfile Output File Format Parameters:=
    End:=

  \endcode

*/

#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/format.h"

int
main(int argc, char* argv[])
{
  USING_NAMESPACE_STIR

  if (argc != 3 && argc != 4)
    {
      std::cerr << "\nUsage: extract_single_images_from_dynamic_image output_filename_pattern input_header_filename "
                   "[output_format_parameter_file]\n\n";
      return EXIT_FAILURE;
    }

  try
    {

      // Read images
      shared_ptr<DynamicDiscretisedDensity> dyn_im_sptr(read_from_file<DynamicDiscretisedDensity>(argv[2]));

      // Check
      if (is_null_ptr(dyn_im_sptr))
        throw std::runtime_error("Failed to read dynamic image (" + std::string(argv[2]) + ").");

      // Set up the output type
      shared_ptr<OutputFileFormat<DiscretisedDensity<3, float>>> output_file_format_sptr;
      if (argc == 3)
        output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr();
      else
        {
          KeyParser parser;
          parser.add_start_key("OutputFileFormat Parameters");
          parser.add_parsing_key("output file format type", &output_file_format_sptr);
          parser.add_stop_key("END");
          std::ifstream in(argv[3]);
          if (!parser.parse(in) || is_null_ptr(output_file_format_sptr))
            throw std::runtime_error("Failed to parse output format file (" + std::string(argv[3]) + ").");
        }

      // Loop over each image
      for (unsigned i = 1; i <= dyn_im_sptr->get_num_time_frames(); ++i)
        {

          DiscretisedDensity<3, float>& disc = dyn_im_sptr->get_density(i);

          std::string current_filename;
          try
            {
              current_filename = format(argv[1], i);
            }
          catch (std::exception& e)
            {
              error(format("Error using 'output_filename' pattern (which is set to '{}'). "
                           "Check syntax for fmt::format. Error is:\n{}",
                           argv[1],
                           e.what()));
              return EXIT_FAILURE;
            }

          // Write to file
          const Succeeded success = output_file_format_sptr->write_to_file(current_filename, disc);
          if (success == Succeeded::no)
            throw std::runtime_error("Failed writing.");
        }

      // If all is good, exit
      return EXIT_SUCCESS;

      // If there was an error
    }
  catch (const std::exception& error)
    {
      std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
      return EXIT_FAILURE;
    }
  catch (...)
    {
      return EXIT_FAILURE;
    }
}
