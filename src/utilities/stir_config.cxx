/*!
  
  Copyright (C) 2021, National Physical Laboratory
  Copyright (C) 2022, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
  \file
  \ingroup utilities
  \brief Prints configuration directory and STIR version
  
  \author Daniel Deidda
  \author Kris Thielemans
  */

#include "stir/find_STIR_config.h" 
#include "stir/error.h"


USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  if(argc==1)
    {
      std::cerr<<"\nUsage: " << "stir_config"
               << " [--config-dir] [--doc-dir] [--examples-dir] [--version]\n\n"
               << "Each option will result in the corresponding text to be written on a separate line,"
               << "(in the same order as the options)\n"
               << "--config-dir:   directory where STIR will read its configuration files.\n"
               << "--doc-dir:      directory with installed STIR documentation.\n"
               << "--examples-dir: directory with installed STIR examples.\n"
               << "--version: version of STIR you are using.\n"<<std::endl;
      exit(EXIT_FAILURE);
    }
  char const * const option=argv[1];
  std::string option_str(option);
    
  while (argc>1)
    {
      if(strcmp(argv[1],"--config-dir")==0)
        std::cout<<get_STIR_config_dir()<<std::endl;
      else if(strcmp(argv[1],"--doc-dir")==0)
        std::cout<<get_STIR_doc_dir()<<std::endl;
      else if(strcmp(argv[1],"--examples-dir")==0)
        std::cout<<get_STIR_examples_dir()<<std::endl;
      else if(strcmp(argv[1],"--version")==0)
        std::cout<<STIR_VERSION_STRING<<std::endl;
      else
        error("Unknown option " + option_str);
      --argc;
      ++argv;
    }
    
}
