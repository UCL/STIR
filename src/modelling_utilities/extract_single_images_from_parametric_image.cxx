//
//
/*
  Copyright (C) 2018, 2020, 2021 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief Split parametric image into individual images
  \author Richard Brown
  \author Kris Thielemans

  \par Usage:
  \code
  extract_single_images_from_parametric_image output_filename_pattern input_header_filename output_format_parameter_file
  \endcode

  The output filename should look something like this: `param_im_{}_output.file_extension`,
  so that we can use fmt::format. In this fashion, you can can specify the output file extension
  should you wish.

  An example of an output parameter file is as follows:
  \code
    OutputFileFormat Parameters:=
    output file format type := interfile
    interfile Output File Format Parameters:=
    number format := float
    number_of_bytes_per_pixel:=4
    End Interfile Output File Format Parameters:=
    End:=

  \endcode

  \sa get_dynamic_images_from_parametric_images.cxx to get dynamic images from parametric images.

*/

#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
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
      std::cerr << "\nUsage: extract_single_images_from_parametric_image output_filename_pattern input_header_filename "
                   "[output_format_parameter_file]\n\n";
      return EXIT_FAILURE;
    }

  try
    {

      // Read images
      auto param_im_sptr(read_from_file<ParametricVoxelsOnCartesianGrid>(argv[2]));

      // Check
      if (is_null_ptr(param_im_sptr))
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
      for (unsigned i = 1; i <= param_im_sptr->get_num_params(); ++i)
        {

          auto disc = param_im_sptr->construct_single_density(i);
          {
            // Get the time frame definition (from start of first frame to end of last)
            ExamInfo exam_info = disc.get_exam_info();
            TimeFrameDefinitions tdefs = exam_info.get_time_frame_definitions();
            const double start = tdefs.get_start_time(1);
            const double end = tdefs.get_end_time(tdefs.get_num_frames());
            tdefs.set_num_time_frames(1);
            tdefs.set_time_frame(1, start, end);
            exam_info.set_time_frame_definitions(tdefs);
            disc.set_exam_info(exam_info);
          }

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
