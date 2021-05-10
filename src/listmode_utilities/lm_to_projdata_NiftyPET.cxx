/*!
  \file 
  \ingroup listmode_utilities
  \ingroup NiftyPET

  \brief Program to bin listmode data to 3d sinograms using STIR's NiftyPET wrapper.

  \author Richard Brown
*/
/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/NiftyPET_listmode/LmToProjDataNiftyPET.h"
#include "stir/ProjData.h"

USING_NAMESPACE_STIR

static void print_usage_and_exit( const char * const program_name, const int exit_status)
{
    std::cerr << "\n\nUsage : " << program_name << " [-h|--help] listmode_binary_file tstart tstop [-N|--norm_binary <filename>] [-p|--prompts <filename>] [-d|--delayeds <filename>] [-r|--randoms <filename>] [-n|--norm_sino <filename>] [--cuda_device <val>] [-v|--verbose <val>]\n\n";
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
        if (argc<4)
            print_usage_and_exit(program_name, EXIT_FAILURE);

        // Get filenames
        const std::string input_filename  = argv[1];
        const int tstart = std::atoi(argv[2]);
        const int tstop = std::atoi(argv[3]);

        // skip past compulsory arguments
        argc-=4;
        argv+=4;

        // Set default value for optional arguments
        std::string p_filename, d_filename, r_filename, input_norm_binary, n_filename;
        char cuda_device(0);
        bool verbose(true);

        // Loop over remaining input
        while (argc>0 && argv[0][0]=='-') {

            if (strcmp(argv[0], "-p")==0 || strcmp(argv[0], "--prompts")==0) {
                p_filename = argv[1];
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "-d")==0 || strcmp(argv[0], "--delayeds")==0) {
                d_filename = argv[1];
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "-r")==0 || strcmp(argv[0], "--randoms")==0) {
                r_filename = argv[1];
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "-N")==0 || strcmp(argv[0], "--norm_binary")==0) {
                input_norm_binary = argv[1];
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "-n")==0 || strcmp(argv[0], "--norm_sino")==0) {
                n_filename = argv[1];
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "--cuda_device")==0) {
                cuda_device = std::atoi(argv[1]);
                argc-=2; argv+=2;
            }
            else if (strcmp(argv[0], "-v")==0 || strcmp(argv[0], "--verbose")==0) {
                verbose = std::atoi(argv[1]);
                argc-=2; argv+=2;
            }
            else {
                std::cerr << "Unknown option '" << argv[0] <<"'\n";
                print_usage_and_exit(program_name, EXIT_FAILURE);
            }
        }
        if (p_filename.empty() && d_filename.empty() && r_filename.empty() && n_filename.empty()) {
            std::cerr << "At least one output filename required.\n";
            print_usage_and_exit(program_name, EXIT_FAILURE);
        }
        if (input_norm_binary.empty() && !n_filename.empty()) {
            std::cerr << "To extract norm sinogram, need to supply norm binary file.\n";
            print_usage_and_exit(program_name, EXIT_FAILURE);
        }

        LmToProjDataNiftyPET lmNP;
        lmNP.set_cuda_device(cuda_device);
        lmNP.set_cuda_verbosity(verbose);
        lmNP.set_listmode_binary_file(input_filename);
        lmNP.set_norm_binary_file(input_norm_binary);
        lmNP.set_start_time(tstart);
        lmNP.set_stop_time(tstop);
        lmNP.process_data();

        // Save outputs
        if (!p_filename.empty()) {
            std::cout << "\n saving prompts sinogram to " << p_filename << "\n";
            lmNP.get_prompts_sptr()->write_to_file(p_filename);
        }
        if (!d_filename.empty()) {
            std::cout << "\n saving delayeds sinogram to " << d_filename << "\n";
            lmNP.get_delayeds_sptr()->write_to_file(d_filename);
        }
        if (!r_filename.empty()) {
            std::cout << "\n saving randoms sinogram to " << r_filename << "\n";
            lmNP.get_randoms_sptr()->write_to_file(r_filename);
        }
        if (!n_filename.empty()) {
            std::cout << "\n saving norm sinogram to " << n_filename << "\n";
            lmNP.get_norm_sptr()->write_to_file(n_filename);
        }
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
