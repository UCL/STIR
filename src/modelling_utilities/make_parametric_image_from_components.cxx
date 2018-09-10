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
  \brief Create parametric image from individual components
  \author Richard Brown

  \par Usage:
  \code 
  make_parametric_image_from_components output_parametric_image slope intercept
  \endcode

  In the case of Patlak, slope is V_0 and intercept is K_i.

*/

#include <iostream>
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"

#include "stir/ProjDataInfo.h"

int main(int argc, char *argv[])
{
    USING_NAMESPACE_STIR

    if (argc != 4) {
        std::cerr << "\nUsage: make_parametric_image_from_components output_parametric_image param1 param2 param3...\n\n";
        std::cerr << "\tCurrently only implemented for 2 kinetic parameters. E.g., for Patlak, slope followed by intercept.\n";
        return EXIT_FAILURE;
    }

    try {

        std::vector<VoxelsOnCartesianGrid<float> > params;

        // Loop over all parameters
        for (int i=2; i<=argc; ++i) {

            // Read
            shared_ptr<DiscretisedDensity<3,float> > im(read_from_file<DiscretisedDensity<3,float> >(argv[i]));
            // Check
            if (is_null_ptr(im)) throw std::runtime_error("Failed to read file: " + std::string(argv[i]) + ".");

            // Convert to VoxelsOnCartesianGrid
            if (is_null_ptr(dynamic_cast<VoxelsOnCartesianGrid<float>*>(im.get())))
                throw std::runtime_error("Failed to convert parameter to VoxelsOnCartesianGrid.");

            VoxelsOnCartesianGrid<float> *param = dynamic_cast<VoxelsOnCartesianGrid<float>*>(im.get());

            params.push_back(*param);

            // Check characteristics match (compare new with first)
            std::string explanation;
            if (!param->has_same_characteristics(params.at(0),explanation))
                throw std::runtime_error("Kinetic images do not have same characteristics (" + std::string(explanation) + ").");
        }

        // At the moment, only implemented for 2 parameters
        if (params.size() == 2) {
            // Construct the parametric image
            ParametricVoxelsOnCartesianGridBaseType base_type(params[0].get_index_range(),params[0].get_origin(),params[0].get_grid_spacing());
            ParametricVoxelsOnCartesianGrid param_im(base_type);

            // Set data
            param_im.update_parametric_image(params[0],1);
            param_im.update_parametric_image(params[1],2);

            // Write it to file
            const Succeeded success = OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->write_to_file(argv[1], param_im);
            if (success == Succeeded::no)
                throw std::runtime_error("Failed writing.");
        }
        else {
            std::cerr << "\ncurrently only implemented for 2 kinetic parameters. Exiting...\n";
            return EXIT_FAILURE;
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


