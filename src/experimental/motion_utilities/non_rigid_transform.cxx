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
  \brief Non-rigid transformation with b-splines
  \author Richard Brown

  \par Usage:
  \code 
  non_rigid_transform output_filename input_filename output_file_format_param displacement_field_4D
  OR
  non_rigid_transform output_filename input_filename output_file_format_param displacement_field_x displacement_field_y displacement_field_z

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
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir_experimental/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "stir_experimental/motion/Transform3DObjectImageProcessor.h"

int main(int argc, char *argv[])
{
    USING_NAMESPACE_STIR

    if (argc < 5 || argc > 8) {
        std::cerr << "\nUsage:\n";
        std::cerr << "\tnon_rigid_transform output_filename input_filename bspline_order displacement_field_4D [output_file_format_param]\n";
        std::cerr << "\t\tOR\n";
        std::cerr << "\tnon_rigid_transform output_filename input_filename bspline_order displacement_field_x displacement_field_y displacement_field_z [output_file_format_param]\n";
        return EXIT_SUCCESS;
    }

    try {

        // Read all the input info
        const std::string output_filename =  argv[1];
        const std::string input_filename  =  argv[2];
        const int         bspline_order   =  atoi(argv[3]);
        std::string disp_4d="", disp_x="", disp_y="", disp_z="";
        if (argc <= 6)
            disp_4d = argv[4];
        else {
            disp_x = argv[4];
            disp_y = argv[5];
            disp_z = argv[6];
        }
        std::string output_file_format_param = "";
        if (argc==6)
            output_file_format_param = argv[5];
        else if (argc==8)
            output_file_format_param = argv[7];

        // Read input image
        std::cerr << "\nReading input image...\n";
        shared_ptr<DiscretisedDensity<3,float> > to_transform_sptr(
                    read_from_file<DiscretisedDensity<3,float> >(input_filename));
        if (is_null_ptr(to_transform_sptr))
            throw std::runtime_error("Failed to read input image (" + input_filename + ").");

        // Create transform
        std::cerr << "\nCreating transform...\n";
        shared_ptr<NonRigidObjectTransformationUsingBSplines<3,float> > fwrd_non_rigid;
        // If 4D
        if (!disp_4d.empty())
            fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(disp_4d,bspline_order));
        else
            fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(disp_x,disp_y,disp_z,bspline_order));

        // Image processor
        std::cerr << "\nDoing transformation...\n";
        Transform3DObjectImageProcessor<float> fwrd_transform(fwrd_non_rigid);
        fwrd_transform.apply(*to_transform_sptr);

        // Save
        std::cerr << "\nSaving result to file (" << output_filename << ")...\n";
        shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format =
                OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
         if (!output_file_format_param.empty()) {
             KeyParser parser;
             parser.add_start_key("output file format parameters");
             parser.add_parsing_key("output file format type", &output_file_format);
             parser.add_stop_key("END");
             if (parser.parse(output_file_format_param.c_str()) == false || is_null_ptr(output_file_format)) {
                warning("Error parsing output file format. Using default format.");
                output_file_format = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
            }
         }
         if (output_file_format->write_to_file(output_filename,*to_transform_sptr) == Succeeded::no)
             throw std::runtime_error("Failed to save to file.");

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


