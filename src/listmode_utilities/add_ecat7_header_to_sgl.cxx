//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for more details.
*/

/*! 
\file
\ingroup utilities
\ingroup ECAT
\brief Prepend contents of ECAT7 header to a new sgl file (from list mode acquisition)
\author Kris Thielemans
\author Nacer Kerrouche
*/


#include "stir/IO/stir_ecat7.h"

#include <string>
#include <stdio.h>
#include <errno.h>

USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7

static void update_main_header(Main_header& mh, const bool is_3d_scan)
  {
    strcpy(mh.study_description, "listmode");
    mh.acquisition_type = DynamicEmission;
    mh.septa_state = 
      is_3d_scan ? SeptaRetracted : SeptaExtended;
    // we set this to a sinogram-type such that header_doc can display the data
    mh.file_type = Short3dSinogram; 
  }


void print_usage_and_exit(const char * const program_name)
  {
    std::cerr<< "\nPrepend contents of ECAT7 header to a sgl file.\n"
	     << "Usage: \n"
	     << "\t" << program_name << " [--2d|--3d] output_sgl_name input_sgl_name input_ECAT7_name \n"
	     << "Defaults to 3D (is used to set septa_state)\n";
    exit(EXIT_FAILURE); 
  }


int main(int argc, char *argv[])
{

  bool is_3d_scan = true;
  const char * const program_name = argv[0];

  if (argc >= 3 && argv[1][0] == '-')
    {
      if (strcmp(argv[1], "--2d") == 0)
	{
	  is_3d_scan = false;
	  --argc; ++argv;
	}
      else if (strcmp(argv[1], "--3d") == 0)
	{
	  is_3d_scan = true;
	  --argc; ++argv;
	}
      else
	print_usage_and_exit(program_name);
    }
  if(argc!=4)
    print_usage_and_exit(program_name);

  const std::string output_name = argv[1];
  const std::string input_name_sgl = argv[2];
  const std::string input_name_ecat7 = argv[3];
 
    {
      FILE * sgl_fptr = fopen(input_name_sgl.c_str(), "rb");
      if (!sgl_fptr) 
	{
	  error("Error opening '%s' for reading: %s", 
		input_name_sgl.c_str(), strerror(errno));
	}
      FILE * out_fptr = fopen(output_name.c_str(), "wb");
      if (!out_fptr) 
	{
	  error("Error opening '%s' for writing: %s", 
		output_name.c_str(), strerror(errno));
	}
      // get ECAT7 header
      Main_header mh_in;
      {
      FILE * ecat7_fptr = fopen(input_name_ecat7.c_str(), "rb");
      if (!ecat7_fptr) 
	{
	  error("Error opening '%s' for reading: %s", 
		input_name_ecat7.c_str(), strerror(errno));
	}
      if (mat_read_main_header(ecat7_fptr, &mh_in)!=0)
	  error("Error reading main header from %s", input_name_ecat7.c_str());
      fclose(ecat7_fptr);
      }

      update_main_header(mh_in, is_3d_scan);
      if (mat_write_main_header(out_fptr, &mh_in))
	    error("Error writing main header to %s", output_name.c_str());
      // copy rest of sgl file into output	

      char buffer[512];
      int success = EXIT_SUCCESS;
      while (!feof(sgl_fptr))
      {
        size_t num_read =
          fread(buffer, 1, 512, sgl_fptr);
        if (ferror(sgl_fptr))
	{
          warning("Error reading '%s' : %s",
		input_name_sgl.c_str(), strerror(errno));
          success = EXIT_FAILURE;
	  break;
        }  
        size_t num_written =
	  fwrite(buffer, 1, num_read, out_fptr);
        if (ferror(sgl_fptr) || num_read!=num_written)
	{
          warning("Error writing '%s' : %s",
		output_name.c_str(), strerror(errno));
          success = EXIT_FAILURE;
	  break;
        }  
      }

      fclose(out_fptr);
      fclose(sgl_fptr);
      return success;
    }

}
