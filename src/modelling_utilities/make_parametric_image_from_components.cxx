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
        std::cerr << "\nUsage: make_parametric_image_from_components output_parametric_image slope intercept\n\n";
        return EXIT_FAILURE;
    }

    try {

        // Read images
        shared_ptr<DiscretisedDensity<3,float> > disc_1(read_from_file<DiscretisedDensity<3,float> >(argv[2]));
        shared_ptr<DiscretisedDensity<3,float> > disc_2(read_from_file<DiscretisedDensity<3,float> >(argv[3]));
        if (is_null_ptr(disc_1))
            throw std::runtime_error("Failed to read dynamic image 1 (" + std::string(argv[2]) + ").");
        if (is_null_ptr(disc_2))
            throw std::runtime_error("Failed to read dynamic image 2 (" + std::string(argv[3]) + ").");

        // Convert them to VoxelsOnCartesianGrid
        if (is_null_ptr(dynamic_cast<VoxelsOnCartesianGrid<float>*>(disc_1.get())))
            throw std::runtime_error("Failed to convert dynamic image 1 to VoxelsOnCartesianGrid.");
        if (is_null_ptr(dynamic_cast<VoxelsOnCartesianGrid<float>*>(disc_2.get())))
            throw std::runtime_error("Failed to convert dynamic image 2 to VoxelsOnCartesianGrid.");
        VoxelsOnCartesianGrid<float> param_1 = *dynamic_cast<VoxelsOnCartesianGrid<float>*>(disc_1.get());
        VoxelsOnCartesianGrid<float> param_2 = *dynamic_cast<VoxelsOnCartesianGrid<float>*>(disc_2.get());

        // Check their characteristics match
        std::string explanation;
        if (!param_1.has_same_characteristics(param_2,explanation))
            throw std::runtime_error("Dynamic images do not have same characteristics (" + std::string(explanation) + ").");

        // Construct the parametric image
        ParametricVoxelsOnCartesianGridBaseType base_type(param_1.get_index_range(),param_1.get_origin(),param_1.get_grid_spacing());
        ParametricVoxelsOnCartesianGrid param_im(base_type);

        // Set data
        param_im.update_parametric_image(param_1,1);
        param_im.update_parametric_image(param_2,2);

        // Write it to file
        const Succeeded success = OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->write_to_file(argv[1], param_im);
        if (success == Succeeded::no)
            throw std::runtime_error("Failed writing.");

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


