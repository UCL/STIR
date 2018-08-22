//
//
/*
  Copyright (C) 2018, University College London
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

/*!
  \file
  \ingroup utilities
  \brief Split dynamic image into individual images
  \author Richard Brown

  \par Usage:
  \code 
  split_dynamic_images output_prefix input output_format
  \endcode

*/

#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"

int main(int argc, char *argv[])
{
    USING_NAMESPACE_STIR

    if (argc != 4) {
        std::cerr <<"\n\nnum args = " << argc << "\n\n\n";
        std::cerr << "\nUsage: split_dynamic_images output_prefix input output_format\n\n";
        return EXIT_FAILURE;
    }

    try {

        // Read images
        shared_ptr<DynamicDiscretisedDensity> dyn_im_sptr(read_from_file<DynamicDiscretisedDensity>(argv[2]));

        // Check
        if (is_null_ptr(dyn_im_sptr))
            throw std::runtime_error("Failed to read dynamic image (" + std::string(argv[2]) + ").");

        // Set up the output type
        shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;
        KeyParser parser;
        parser.add_start_key("Test OutputFileFormat Parameters");
        parser.add_parsing_key("output file format type", &output_file_format_sptr);
        parser.add_stop_key("END");
        std::ifstream in(argv[3]);
        if (!parser.parse(in))
            throw std::runtime_error("Failed to parse output format file (" + std::string(argv[3]) + ").");
        if (is_null_ptr(output_file_format_sptr))
            throw std::runtime_error("Failed to parse output format file (" + std::string(argv[3]) + ").");

        // Loop over each image
        for (unsigned i=1; i<=dyn_im_sptr->get_num_time_frames(); ++i) {

            DiscretisedDensity<3,float> &disc = dyn_im_sptr->get_density(i);

            // Get filename
            std::ostringstream filename;
            filename << argv[1] << "_" << i;

            // Write to file
            const Succeeded success = output_file_format_sptr->write_to_file(filename.str(),disc);
            if (success == Succeeded::no)
                throw std::runtime_error("Failed writing.");
        }

        // If all is good, exit
        return EXIT_SUCCESS;

    // If there was an error
    } catch(const std::exception &error) {
        std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
        return EXIT_FAILURE;
    } catch(...) {
        return EXIT_FAILURE;
    }
}


