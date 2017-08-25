/*!
  \file
  \ingroup utilities
  \ingroup recon_buildblock
  \brief Demo for Realtime reconstruction initialization

  \author Nikos Efthimiou


  \par main() for any reconstruction which will be created in realtime.
  \code
  recontest parfile
  \endcode

*/


#include "stir/DiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/Reconstruction.h"
#include <iostream>
#include <stdlib.h>
#include <string>
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/IO/write_to_file.h"

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

    std::string output_filename;

    KeyParser parser;
    parser.add_start_key("Reconstruction");
    parser.add_stop_key("End Reconstruction");
    parser.add_key("output filename prefix", &output_filename);
    parser.add_parsing_key("reconstruction method", &reconstruction_method_sptr);
    parser.parse(argv[1]);

    HighResWallClockTimer t;
    t.reset();
    t.start();


    if (reconstruction_method_sptr->reconstruct() == Succeeded::yes)
    {
        t.stop();
        std::cout << "Total Wall clock time: " << t.value() << " seconds" << std::endl;
    }
    else
    {
        t.stop();
        return Succeeded::no;
    }

    //
    // Save the reconstruction output from this location.
    //

    if (output_filename.length() > 0 )
    {
        shared_ptr  < DiscretisedDensity < 3, float > > reconstructed_image =
                reconstruction_method_sptr->get_target_image();

        OutputFileFormat<DiscretisedDensity < 3, float > >::default_sptr()->
                write_to_file(output_filename, *reconstructed_image.get());
    }

    return Succeeded::yes;

    return EXIT_SUCCESS;
}

