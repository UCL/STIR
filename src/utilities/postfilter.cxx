//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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

  \brief  This program performs filtering on image data
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project
  

  This program enables calling any ImageProcessor object on input data, 
  and writing it to file. It can take the following command line:
  \verbatim
   postfilter [[-verbose] <output filename > <input header file name> <filter .par filename>
  \endverbatim
  This is done to make it easy to process a lot of files with the same 
  ImageProcessor. However, if the number of command line arguments is not 
  correct, appropriate questions will be asked interactively.

  If the <tt>--verbose</tt> option is used, the filter-parameters that are going to be
  used will be written to stdout. This is useful for checking/debugging.

  \par Example .par file
  (but see stir::MedianImageFilter3D to see if the following example is still correct)
  \verbatim
  PostFilteringParameters :=
  Postfilter type :=Median   
  Median Filter Parameters :=
  mask radius x := 1   
  mask radius y := 2
  mask radius z := 3
  End Median Filter Parameters:=
  End PostFiltering Parameters:=
  \endverbatim

*/

#include "stir/PostFiltering.h"
#include "stir/utilities.h"
#include "stir/DiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

DiscretisedDensity<3,float>* ask_image(const char *const input_query)
{
  
  char filename[max_filename_length];
  ask_filename_with_extension(filename, 
				input_query,
				"");
  
  return DiscretisedDensity<3,float>::read_from_file(filename);
  
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

static void
print_usage()
{
  cerr<<"\nUsage: postfilter [--verbose] <output filename > <input header file name> <filter .par filename>\n"<<endl;
}

int
main(int argc, char *argv[])
{
  
  shared_ptr<DiscretisedDensity<3,float> > input_image_ptr;
  PostFiltering<DiscretisedDensity<3,float> > post_filtering;
  std::string out_filename;
  bool verbose = false;

  // option processing
  if (argc>1 && argv[1][0] == '-')
    {
      if (strcmp(argv[1], "--verbose") == 0)
	{
	  verbose = true;
	  --argc; ++argv;
	}
      else
	{
	  print_usage();
	  return EXIT_FAILURE;
	}
    }

  if (argc!=4)
    {
      print_usage();
    }
  if (argc>1)
    {
      out_filename = argv[1];
    }
  else
    {
      char outfile[max_filename_length];
      ask_filename_with_extension(outfile,
				  "Output to which file: ", "");
      out_filename = outfile;
    }
  if (argc>2)
    {
      input_image_ptr = 
	read_from_file<DiscretisedDensity<3,float> >(argv[2]);
    }
  else
    {
      input_image_ptr.reset(ask_image("Image to process?"));
    }
  if (argc>3)
    {
      if (post_filtering.parse(argv[3]) == false)
	{
	  warning("postfilter aborting because error in parsing. Not writing any output");
	  return EXIT_FAILURE;
	}
    }
  else
    {     
      cerr << "\nI'm going to ask you for the type of filter (or image processor)\n"
	"Possible values:\n";
      DataProcessor<DiscretisedDensity<3,float> >::list_registered_names(cerr);
      
      post_filtering.ask_parameters();
    }

  if (post_filtering.is_filter_null())
    {
      warning("postfilter: No filter set. Not writing any output.\n");
      return EXIT_FAILURE;
    }

  if (is_null_ptr(input_image_ptr))
    {
      warning("postfilter: No input image. Not writing any output.\n");
      return EXIT_FAILURE;
    }
    
  if (verbose)
    {
      cerr << "PostFilteringParameters:\n" << post_filtering.parameter_info();
    }

  if (post_filtering.process_data(*input_image_ptr) == Succeeded::no)
      return EXIT_FAILURE;
  
  if (OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file(out_filename,*input_image_ptr) == Succeeded::yes)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}


