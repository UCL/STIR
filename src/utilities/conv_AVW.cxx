//
//
/*
    Copyright (C) 2001- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief convert image files to different file format. Images are read using the AVW library.
\author Kris Thielemans 

*/

#include "stir/IO/stir_AVW.h"
#include "AVW_ImageFile.h"

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/utilities.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include <iostream>
#include <string.h>

void print_usage_and_exit( const char * const program_name)
{
  {
    std::cerr << "Usage : " << program_name << " [ --flip_z ] imagefile\n";

    AVW_List* list = AVW_ListFormats(AVW_SUPPORT_READ);
    std::cerr << "Supported file formats for reading by AVW:\n ";
    for (int i=0; i<list->NumberOfEntries; ++i)
      { 
	std::cerr << list->Entry[i] << '\n';
    }
    exit(EXIT_FAILURE);
  }
}

int
main(int argc, char **argv)
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool flip_z = false;

  // first process command line options
  while (argc>0 && argv[0][0]=='-')
  {
    if (strcmp(argv[0], "--flip_z")==0)
      {
	flip_z = true;
	argc-=1; argv+=1;
      }
    else
    { 
      std::cerr << "Unknown option '" << argv[0] <<"'\n"; 
      print_usage_and_exit(program_name); 
    }
  }

  if (argc != 1)
      print_usage_and_exit(program_name); 

  char *imagefile = argv[0];
  stir::shared_ptr<stir::OutputFileFormat<stir::DiscretisedDensity<3,float> > > 
    output_file_format_sptr =
    stir::OutputFileFormat<stir::DiscretisedDensity<3,float> >::default_sptr();
  {
    // open non-existent file first
    // this is necessary to get AVW_LoadObjectMap to work
    AVW_ImageFile *avw_file= AVW_OpenImageFile("xxx non-existent I hope","r");
  }
  
  std::cout << "Reading ImageMap " << imagefile << '\n';
  AVW_ImageFile*avw_file = AVW_OpenImageFile(imagefile,"r");
  if(!avw_file) 
    { 
      AVW_Error("AVW_OpenImageFile"); 
      std::cout << std::flush;
      exit(EXIT_FAILURE); 
    }
  
  std::cout << "Number of volumes: " << avw_file->NumVols << '\n';
  unsigned int min_volume_num = 0;
  unsigned int max_volume_num = avw_file->NumVols-1;
  if (!stir::ask("Attempt all data-sets (Y) or single data-set (N)", true))
    {
      min_volume_num = max_volume_num =
	stir::ask_num("Volume number ? ", 1U, avw_file->NumVols, 1U)
	- 1; // subtract 1 as AVW numbering starts from 0
    }
  {
    AVW_Volume *volume = NULL;
    for (unsigned int volume_num=min_volume_num; volume_num<=max_volume_num; ++volume_num)
    {
      {
  	volume = AVW_ReadVolume(avw_file, volume_num, volume);
	if(!volume) 
	  { 
	    AVW_Error("AVW_ReadVolume"); 
	    stir::warning("Error in volume %d. Skipping...", volume_num+1);//, AVW_ErrorMessage);
	    continue; 
	  }
        stir::shared_ptr<stir::VoxelsOnCartesianGrid<float> > stir_volume_sptr =
          stir::AVW::AVW_Volume_to_VoxelsOnCartesianGrid(volume, flip_z);
	if (stir::is_null_ptr(stir_volume_sptr))
	  { 
	    stir::warning("Error converting volume to STIR format. Skipping...", volume_num);
	    continue;
	  }
	char *header_filename = new char[strlen(imagefile) + 100];
	{
	  strcpy(header_filename, imagefile);
	  // keep extension, just in case we would have conflicts otherwise
	  // but replace the . with a _
	  char * dot_ptr = strchr(stir::find_filename(header_filename),'.');
	  if (dot_ptr != NULL)
	    *dot_ptr = '_';
	  // now add stuff to say which frame, gate, bed, data this was
	  sprintf(header_filename+strlen(header_filename), "_f%ug%dd%db%d",
		  volume_num+1, 1, 0, 0);
	}
	if (output_file_format_sptr->write_to_file(header_filename, *stir_volume_sptr)
	    == stir::Succeeded::no)
	  {
	    stir::warning("Error writing %s", header_filename);
	  }
	else
	  {
	    std::cout << "Wrote " << header_filename << '\n';
	  }
	delete[] header_filename;
      }
    }
    AVW_DestroyVolume(volume);
  }
  AVW_CloseImageFile(avw_file);

	
  return(EXIT_SUCCESS);
}


