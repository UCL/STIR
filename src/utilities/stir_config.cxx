/*!
  
  Copyright (C) 2021, National Physical Laboratory
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
  \file
  \ingroup utilities
  \brief Prints configuration directory and STIR version
  
  \author Daniel Deidda
  */

#include "stir/find_STIR_config.h" 
#include "stir/error.h"


USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
    if(argc!=3 && argc!=2)
    {
      std::cerr<<"\nUsage: " << "stir_config_dir" 
               << " [--config-dir] [--version] \n";
      std::cerr << "Use the option --config-dir to output the STIR config directory.\n"
       << "Use the option --version to output the version of STIR you are using.\n"<<std::endl;
      exit(EXIT_FAILURE);
    }
    bool print_dir, print_version;
    char const * const option=argv[1];
    std::string option_str(option);
    
    while (argc>1 && strncmp(argv[1],"--",2)==0)
    {
          if(strcmp(argv[1],"--config-dir")==0)
              print_dir=true;
          else if(strcmp(argv[1],"--version")==0)
              print_version=true;
          else
              error("Unknown option " + option_str);
          --argc;
          ++argv;
    }
    
    
    if (print_dir)
        std::cout<<"Config directory is "<<get_STIR_config_dir()<<std::endl;
    if (print_version)
        std::cout<<"You are using STIR "<<STIR_VERSION_STRING<<std::endl;
}
