/*!
  
  Copyright (C) 2021, National Physical Laboratory
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
  \file
  \ingroup utilities
  \brief Print configuration directory
  
  \author Daniel Deidda
  */

#include "stir/find_STIR_config.h" 
#include "stir/error.h"


USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
    bool print_dir, print_version;
    char const * const option=argv[1];
    std::string option_str(option);
    int argc_num=argc;
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
    
    if(argc_num!=3 && argc_num!=2)
    {
      std::cerr<<"\nUsage: " << "stir_config_dir" 
               << " [--config-dir] [--version] \n";
      std::cerr << "Use the option --config-dir to output the STIR config directory.\n"
       << "Use the option --version to output the version of STIR you are using.\n"<<std::endl;
      exit(EXIT_FAILURE);
    }
    
    if (print_dir)
        find_STIR_config_dir();
    if (print_version)
        std::cout<<"You are using STIR "<<STIR_VERSION_STRING<<std::endl;
}
