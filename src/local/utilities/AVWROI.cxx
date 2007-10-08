//
// $Id$
//
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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
\brief convert AVW ROI files to images. Images are read using the AVW library.
\author Kris Thielemans 

$Date$
$Revision$ 
*/
#include "stir/IO/stir_AVW.h"
#include "AVW_ObjectMap.h"
#include "AVW_ImageFile.h"

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/utilities.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include <iostream>

USING_NAMESPACE_STIR

void print_usage_and_exit( const char * const program_name)
{
  {
    std::cerr << "Usage : " << program_name << " [ --flip_z ] Analyzeobjectfile.obj\n";

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

  char *objectfile = argv[0];
  stir::shared_ptr<stir::OutputFileFormat<stir::DiscretisedDensity<3,float> > > 
    output_file_format_sptr =
    stir::OutputFileFormat<stir::DiscretisedDensity<3,float> >::default_sptr();
  {
    // open non-existent file first
    // this is necessary to get AVW_LoadObjectMap to work
    AVW_ImageFile *avw_file= AVW_OpenImageFile("xxx non-existent I hope","r");
  }
  
  AVW_ObjectMap *object_map = AVW_LoadObjectMap(objectfile);
  if(!object_map) { AVW_Error("AVW_LoadObjectMap"); exit(EXIT_FAILURE); }

  std::cerr << "Number of objects: " << object_map->NumberOfObjects << '\n';  
  {
    AVW_Volume *volume = NULL;
    for (int object_num=0; object_num<object_map->NumberOfObjects; ++object_num)
    {
      const char * const object_name = object_map->Object[object_num]->Name;
      std::cerr << "Object " <<  object_num << ": " << object_name << '\n';
          
      if (ask("Write this one?",true))
      {
        volume = AVW_GetObject(object_map, object_num, volume);
	if(!volume) 
	  { 
	    AVW_Error("AVW_GetObject"); 
	    stir::warning("Error in object. Skipping...");//, AVW_ErrorMessage);
	    continue; 
	  }
	
        shared_ptr<VoxelsOnCartesianGrid<float> > stir_volume_sptr =
          stir::AVW::AVW_Volume_to_VoxelsOnCartesianGrid(volume, flip_z);
	if (stir::is_null_ptr(stir_volume_sptr))
	  { 
	    stir::warning("Error converting object to STIR format. Skipping...", object_num);
	    continue;
	  }
	char *header_filename = new char[strlen(objectfile) + strlen(object_name) + 10];
	{
	  strcpy(header_filename, objectfile);
	  // append object_name, but after getting rid of the extension in objectfile
	  replace_extension(header_filename, "_");
	  strcat(header_filename, object_name);
	}
	warning("Setting voxel size to 962 defaults\n");
	stir_volume_sptr->set_voxel_size(Coordinate3D<float>(2.425F,2.25F,2.25F));
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
  AVW_DestroyObjectMap(object_map);

  return(EXIT_SUCCESS);
}


