//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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
  \ingroup scatter
  \brief Estimates a coarse scatter sinogram

  \author Kris Thielemans
  
        
  \par Usage:
  \code
  estimate_scatter parfile
  \endcode
  See stir::ScatterEstimationByBin documentation for the format
  of the parameter file.
*/

#include "stir/scatter/ScatterEstimation.h"
#include "stir/Succeeded.h"
/***********************************************************/     

static void print_usage_and_exit()
{
    std::cerr<<"This executable runs a Scatter simulation method based on the options "
               "in a parameter file";
    std::cerr<<"\nUsage:\n simulate_scatter scatter_simulation.par\n";
    std::cerr<<"Example parameter file:\n\n"
               "Scatter Estimation Parameters :=\n"
               ";Run in debug mode\n"
               ";; A new folder called extras will be created, in which many\n"
               ";; extra files will be stored \n"
               "run in debug mode := 1\n"
              " recompute initial activity image := 1\n"
               "initial activity image filename := \n"

               "zoom xy := 0.3\n"
               "zoom z := 1.0\n"
               "input file := \n"
               "recompute attenuation projdata := 0\n"
               "αttenuation projdata filename := \n"
               "attenuation image filename := \n"
               "normalisation coefficients filename := \n"
               "recompute mask projdata := \n "
               "mask projdata filename := \n "

               "recompute mask image :=\n "
               "mask image filename := \n"

               "mask max threshold :=\n"
               "mask add scaler := \n "
               "mask min threshold :=\n"
               "mask times scalar := \n"
               "mask prostfilter filename :=\n"
               "; End of Mask\n"

               "tail fitting par filename :=\n "

               ";Backgroud data\n"
               "background projdata filename := \n"
               "; export SSRB sinograms\n"
               "; export 2d projdata := \n"

               "; ScatterSimulation Stuff \n"
               "scatter simulation parameter file :=\n"

               "; Over-ride the values set in the scatter simulation parameteres file\n"
               "over-ride initial activity image := \n"
               "over-ride density image := \n "
               "over-ride density image for scatter points := \n"

               "reconstruction template file := \n"
               "; This is the number of times which the Scatter Estimation will\n"
               "; iterate. Default is 5\n"

               "number of scatter iterations := \n"

               "; Average the first two activity images\n"
               "do average at 2 := \n"

               "; Export scatter estimates of each iteration \n"
               "export scatter estimates of each iteration := \n"

               "output scatter estimate name prefix := \n"

               "; If provided it will be given to the ScatterSimulation \n"
               "; subsampled attenuation imagw filename := \n"

               "max scale value := \n"
               "min scale value := \n"

               ";Upsample and fit \n"
               "; defaults to 3.\n"
               "half filter width := \n"
               "remove interleaving := \n"

               "End Scatter Estimation Parameters :=" << std::endl;
               exit(EXIT_FAILURE);
}
/***********************************************************/

int main(int argc, const char *argv[])                                  
{         
    stir::ScatterEstimation scatter_estimation;

    if (argc==2)
    {
        if (scatter_estimation.parse(argv[1]) == false)
            return EXIT_FAILURE;
    }
    else
        print_usage_and_exit();

    return scatter_estimation.process_data() == stir::Succeeded::yes ?
                EXIT_SUCCESS : EXIT_FAILURE;
}

