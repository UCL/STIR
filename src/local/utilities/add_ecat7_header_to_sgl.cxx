//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    For internal use only
*/

/*! 
\file
\ingroup utilities
\ingroup ECAT
\brief Prepend contents of ECAT7 header to a new sgl file (from list mode acquisition)
\author Kris Thielemans
\author Nacer Kerrouche
$Date$
$Revision$
*/


#include "stir/IO/stir_ecat7.h"

#include <string>
#include <stdio.h>


USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7

static void update_main_header(Main_header& mh)
  {
    strcpy(mh.study_description, "listmode");
    mh.acquisition_type = DynamicEmission;
    mh.septa_state = SeptaRetracted; // TODO get from acquisition script
    mh.file_type = 0;
  }




int main(int argc, char *argv[])
{


  if(argc!=4)
  {
    std::cerr<< "\nPrepend contents of ECAT7 header to a sgl file.\n"
        << "Usage: \n"
	<< "\t" << argv[0] << "  output_sgl_name input_sgl_name input_ECAT7_name \n";
    return EXIT_FAILURE; 
  }

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

      update_main_header(mh_in);
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
