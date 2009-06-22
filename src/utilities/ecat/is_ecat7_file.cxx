//
// $Id$
//
/*
    Copyright (C) 2007- $Date$, Hammersmith Imanet Ltd
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
  \ingroup ECAT
  \brief A simple utility that allows to check if a file is in the ECAT7 format.

  \author Kris Thielemans

  $Date$
  $Revision$

  \par Usage
  \code
   is_ecat7_file [--image | --emission | --ACF ] in_name 
  \endcode
  Without options, it will just check if it's an ECAT7 file, otherwise the type 
  is checked as well.

  The utilities will either print yes or no to stdout and return \c EXIT_SUCCESS,
  or print a warning and return \c EXIT_FAILURE otherwise.

*/
#include "stir/IO/stir_ecat7.h"

USING_NAMESPACE_STIR

static void print_usage_and_exit(const char * prog_name)
{
  std::cerr << "Usage: " << prog_name << " [--image | --emission | --ACF ] in_name \n";
  std::cerr << "Will print yes or no (or return an error if the command line syntax was wrong)\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  const char * prog_name = argv[0];
  
  enum options_type { opt_image, opt_emission, opt_ACF, opt_none};
  options_type option = opt_none;

  while (argc>1 && argv[1][0] == '-')
    {
      if (strcmp(argv[1], "--image") == 0)
	{
	  option = opt_image; --argc; ++argv; break;
	}
      if (strcmp(argv[1], "--emission") == 0)
	{
	  option = opt_emission; --argc; ++argv; break;
	}
      if (strcmp(argv[1], "--ACF") == 0)
	{
	  option = opt_ACF; --argc; ++argv; break;
	}
      else
	print_usage_and_exit(prog_name);
    }
  if (argc!=2)
    print_usage_and_exit(prog_name);

  const char * filename = argv[1];

  bool value;
  switch (option)
    {
    case opt_none: value = stir::ecat::ecat7::is_ECAT7_file(filename); break;
    case opt_image: value = stir::ecat::ecat7::is_ECAT7_image_file(filename); break;
    case opt_emission: value = stir::ecat::ecat7::is_ECAT7_emission_file(filename); break;
    case opt_ACF: value = stir::ecat::ecat7::is_ECAT7_attenuation_file(filename); break;
    default: value=false;
    }

  if (value)
    std::cout << "yes\n";
  else
    std::cout << "no\n";
  return EXIT_SUCCESS;
}
