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
  \ingroup recon_buildblock
  \brief Demo for Realtime reconstruction initialization

  \author Nikos Efthimiou


  \par main() for General_Reconstruction demo
  \code
  Reconstruct_test parfile
  \endcode

*/


#include "stir/DiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/Reconstruction.h"
#include <iostream>
#include <stdlib.h>
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"

static void print_usage_and_exit()
{
    std::cerr<<"This executable is able to reconstruct some data without calling a specific reconstruction method, from the code.\n";
    std::cerr<<"but specifing the method in the par file with the \"econstruction method\". \n";
    std::cerr<<"\nUsage:\nrecontest reconstuction.par\n";
    std::cerr<<"Example parameter file:\n\n"
            <<"reconstruction method := OSMAPOSL\n"
           <<"OSMAPOSLParameters := \n"
          <<"objective function type:= PoissonLogLikelihoodWithLinearModelForMeanAndProjData\n"
         <<"PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=\n"
        <<"input file := <input>.hs\n"
       <<"maximum absolute segment number to process := -1\n"
      <<"projector pair type := Matrix\n"
     <<"Projector Pair Using Matrix Parameters :=\n"
    <<"Matrix type := Ray Tracing\n"
    <<"Ray tracing matrix parameters :=\n"
    <<"number of rays in tangential direction to trace for each bin:= 10\n"
    <<"End Ray tracing matrix parameters :=\n"
    <<"End Projector Pair Using Matrix Parameters :=\n"
    <<"recompute sensitivity := 1\n"
    <<"zoom := 1\n"
    <<"end PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=\n"
    <<"enforce initial positivity condition:= 1 \n"
    <<"number of subsets:= 1\n"
    <<"number of subiterations:= 1 \n"
    <<"save estimates at subiteration intervals:= 1\n"
    <<"output filename prefix := output\n"
    <<"end OSMAPOSLParameters := \n"
    <<"end reconstruction := \n";
      exit(EXIT_FAILURE);
}


/***********************************************************/

int main(int argc, const char *argv[])
{
    using namespace stir;

    if (argc!=2)
        print_usage_and_exit();

    shared_ptr < Reconstruction < DiscretisedDensity < 3, float > > >
            reconstruction_method_sptr;

    KeyParser parser;
    parser.add_start_key("Reconstruction");
    parser.add_stop_key("End Reconstruction");
    parser.add_parsing_key("reconstruction method", &reconstruction_method_sptr);
    parser.parse(argv[1]);

    HighResWallClockTimer t;
    t.reset();
    t.start();

    if (reconstruction_method_sptr->reconstruct() == Succeeded::yes)
    {
        t.stop();
        std::cout << "Total Wall clock time: " << t.value() << " seconds" << std::endl;
        return Succeeded::yes;
    }
    else
    {
        t.stop();
        return Succeeded::no;
    }



    return EXIT_SUCCESS;

}

