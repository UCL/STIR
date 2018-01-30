/*
  Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \brief Compute a mask for the "tails" in the sinogram

  As from 23 July 2016, the functionality of this executable was transfered in
  a new class CreateTailMaskFromACFs. This made possible to use par file to
  initiliase the process and use it from within some other code.

  \author Nikos Efthimiou
  \author Kris Thielemans
	
  \par Usage:

  \verbatim
   create_tail_mask_from_ACFs --ACF-filename <filename> \\
        --output-filename <filename> \\
        [--ACF-threshold <float>] \\
        [--safety-margin <integer>]
  \endverbatim

  \par Alternative Usage:

  \verbatim
    create_tail_mask_from_ACFs <filename.par>
  \endverbatim

  \par Example of parameter file:
  \verbatim
    CreateTailMaskFromACFs :=
        ACF-filename :=
        output-filename :=
        ACF-threshold :=
        safety-margin :=
    END CreateTailMaskFromACFs :=
  \endverbatim
  ACF-threshold defaults to 1.1 (should be larger than 1), safety-margin to 4
*/

#include <iostream>
#include <fstream>
#include <string>

#include "stir/scatter/CreateTailMaskFromACFs.h"


/***********************************************************/     

static void
print_usage_and_exit(const char * const prog_name)
{
    std::cerr << "\nUsage:\n" << prog_name << "\n"
              << "\t--ACF-filename <filename>\n"
              << "\t--output-filename <filename>\n"
              << "\t[--ACF-threshold <float>]\n"
              << "\t[--safety-margin <integer>]\n"
              << "ACF-threshold defaults to 1.1, safety-margin to 4\n"
              << "Alternative Usage:\n"
              << "Create_tail_mask_from_ACFs parameters.par\n"
              << "Example par file:";
    exit(EXIT_FAILURE);
}

int main(int argc, const char *argv[])                                  
{         
    USING_NAMESPACE_STIR;
    const char * const prog_name = argv[0];

    CreateTailMaskFromACFs create_tail_mask_from_ACFs;



    // If one arg is supplied and it is a par file
    // then use the Create_tail_mask_from_ACFs to parse the
    // file. Otherwise continue the old way.
    if (argc == 2)
    {
        std::stringstream hdr_stream(argv[1]);
        std::string sargv = hdr_stream.str();

        size_t lastindex = sargv.find_last_of(".");
        std::string extension = sargv.substr(lastindex);
        std::string par_ext = ".par";
        if ( extension.compare(par_ext) != 0 )
            error("Please provide a valid par file.");

        if (create_tail_mask_from_ACFs.parse(argv[1]) == false)
        {
            warning("Create_tail_mask_from_ACFs aborting because error in parsing. Not writing any output");
            return EXIT_FAILURE;
        }
    }
    else
    {
        // option processing
        float ACF_threshold = 1.1F;
        int safety_margin=4;
        std::string ACF_filename;
        std::string output_filename;

        while (argc>2 && argv[1][1] == '-')
        {
            if (strcmp(argv[1], "--ACF-filename")==0)
            {
                ACF_filename = (argv[2]);
                argc-=2; argv +=2;
            }
            else if (strcmp(argv[1], "--output-filename")==0)
            {
                output_filename = (argv[2]);
                argc-=2; argv +=2;
            }
            else if (strcmp(argv[1], "--ACF-threshold")==0)
            {
                ACF_threshold = float(atof(argv[2]));
                argc-=2; argv +=2;
            }
            else if (strcmp(argv[1], "--safety-margin")==0)
            {
                safety_margin = atoi(argv[2]);
                argc-=2; argv +=2;
            }
            else
            {
                std::cerr << "\nUnknown option: " << argv[1];
                print_usage_and_exit(prog_name);
            }
        }

        if (argc!=1 || ACF_filename.size()==0 || output_filename.size()==0)
        {
            print_usage_and_exit(prog_name);
        }

        // Use CreateTailMaskFromACFs::set_up to parse the parameters
        create_tail_mask_from_ACFs.set_input_projdata(ACF_filename);
        create_tail_mask_from_ACFs.set_output_projdata(output_filename);
        create_tail_mask_from_ACFs.ACF_threshold = ACF_threshold;
        create_tail_mask_from_ACFs.safety_margin = safety_margin;

    }

    // Onwards the new class will do the job ...

    return create_tail_mask_from_ACFs.process_data() == stir::Succeeded::yes ?
                EXIT_SUCCESS : EXIT_FAILURE;
}                 
     
