/*!
  \file 
  \ingroup listmode_utilities

  \brief Program to bin listmode data to 3d sinograms using STIR's NiftyPET wrapper.

  \author Richard Brown
*/
/*
    Copyright (C) 2020, University College London
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

#include "stir/listmode/niftypet_listmode/LmToProjDataNiftyPET.h"
#include "stir/ProjData.h"

USING_NAMESPACE_STIR

static void print_usage_and_exit( const char * const program_name, const int exit_status)
{
    std::cerr << "\n\nUsage : " << program_name << " [-h|--help] output_filename listmode_binary_file tstart tstop [--cuda_device <val>]\n\n";
    exit(exit_status);
}

int main(int argc, char * argv[])
{
    try {
        const char * const program_name = argv[0];

        // Check for help request
        for (int i=1; i<argc; ++i)
            if (strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--help")==0)
                print_usage_and_exit(program_name, EXIT_SUCCESS);

        // Check for all compulsory arguments
        if (argc<5)
            print_usage_and_exit(program_name, EXIT_FAILURE);

        // Get filenames
        const std::string output_filename = argv[1];
        const std::string input_filename  = argv[2];
        const int tstart = std::atoi(argv[3]);
        const int tstop = std::atoi(argv[4]);

        // skip past compulsory arguments
        argc-=5;
        argv+=5;

        // Set default value for optional arguments
        char cuda_device(0);

        // Loop over remaining input
        while (argc>0 && argv[0][0]=='-') {
            if (strcmp(argv[0], "--cuda_device")==0) {
            cuda_device = std::atoi(argv[1]);
                argc-=1; argv+=1;
            }
            else {
                std::cerr << "Unknown option '" << argv[0] <<"'\n";
                print_usage_and_exit(program_name, EXIT_FAILURE);
            }
        }

        LmToProjDataNiftyPET lmNP;
        lmNP.set_cuda_device(cuda_device);
        lmNP.set_listmode_binary_file(input_filename);
        lmNP.set_start_time(tstart);
        lmNP.set_stop_time(tstop);
        lmNP.process_data();

        // Save output
        lmNP.get_output()->write_to_file(output_filename);
    }

    // If there was an error
    catch(const std::exception &error) {
        std::cerr << "\nError encountered:\n\t" << error.what() << "\n\n";
        return EXIT_FAILURE;
    }
    catch(...) {
        std::cerr << "\nError encountered.\n\n";
        return EXIT_FAILURE;
    }
    return(EXIT_SUCCESS);
}
