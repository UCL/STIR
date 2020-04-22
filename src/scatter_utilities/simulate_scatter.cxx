//
//
/*
  Copyright (C) 2016, UCL
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
  \brief Simulates a coarse scatter sinogram

  \author Nikos Efthimiou

  \par Usage:
  \code
  simulate_scatter parfile
  \endcode
  See stir::ScatterSimulation documentation for the format
  of the parameter file.
*/

#include "stir/scatter/ScatterSimulation.h"
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"

using std::cerr;
using std::cout;
using std::endl;

static void print_usage_and_exit()
{
    std::cerr<<"This executable runs a Scatter simulation method based on the options "
               "in a parameter file";
    std::cerr<<"\nUsage:\n simulate_scatter scatter_simulation.par\n";
    std::cerr<<"Example parameter file:\n\n"
               "Scatter Simulation :=\n"
               "Simulation method := Single Scatter Simulation\n"
               "Scatter Simulation Parameters :=\n"
               " template projdata filename :=\n"
               "attenuation image filename := \n"
               "attenuation image for scatter points filename := \n"
               "activity image filename :=\n"
               "output filename prefix := \n"
               "zoom XY for attenuation image for scatter points := 1\n"
               "zoom Z for attenuation image for scatter points: := 1\n"
               "XY size of downsampled image for scatter points :=\n"
               "Z size of downsampled image for scatter points :=\n"
               "attenuation image for scatter points output filename :=\n"
               "downsampled scanner number of detectors per ring := 32\n"
               "downsampled scanner number of rings := \n"
               "attenuation threshold := 0.01\n"
               "random := 1\n"
               "use cache := 1\n"
               "End Scatter Simulation Parameters :=\n"
               "End Scatter Simulation:="<< std::endl;
               exit(EXIT_FAILURE);
}
/***********************************************************/

int main(int argc, const char *argv[])
{
    USING_NAMESPACE_STIR

    HighResWallClockTimer t;
    t.reset();
    t.start();

    if (argc!=2)
        print_usage_and_exit();
    shared_ptr < ScatterSimulation >
            simulation_method_sptr;
    KeyParser parser;
    parser.add_start_key("Scatter Simulation");
    parser.add_stop_key("End Scatter Simulation");
    parser.add_parsing_key("Simulation method", &simulation_method_sptr);
    if (!parser.parse(argv[1]))
      { t.stop(); return EXIT_FAILURE; }

    if(simulation_method_sptr->set_up() == Succeeded::no)
      { t.stop(); return EXIT_FAILURE; }

    if(simulation_method_sptr->process_data() == stir::Succeeded::yes)
    {
        t.stop();
        cout << "Total Wall clock time: " << t.value() << " seconds" << endl;
        return EXIT_SUCCESS;
      }
    else
      {
        t.stop();
        return EXIT_FAILURE;
      }
}

