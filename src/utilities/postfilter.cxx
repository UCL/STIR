//
// $Id$
//

/*!
  \file 
  \ingroup utilities

  \brief  This program performs filtering on image data
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project
  
  $Date$
  $Revision$

  This program enables calling any ImageProcessor object on input data, 
  and writing it to file. It can take the following command line:
  \verbatim
   postfilter <output filename > <input header file name> <filter .par filename>
  \endverbatim
  This is done to make it easy to process a lot of files with the same 
  ImageProcessor. However, if the number of command line arguments is not 
  correct, appropriate questions will be asked interactively.

  \par Example .par file
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
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/utilities.h"
#include "stir/KeyParser.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ImageProcessor.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/is_null_ptr.h"

#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

DiscretisedDensity<3,float>* ask_image(char *input_query)
{
  
  char filename[max_filename_length];
  ask_filename_with_extension(filename, 
				input_query,
				"");
  
  return DiscretisedDensity<3,float>::read_from_file(filename);
  
}

class PostFiltering 
{
public:
  PostFiltering();
  shared_ptr<ImageProcessor<3,float> > filter_ptr;
public:
  KeyParser parser;
  
};

PostFiltering::PostFiltering()
{
  filter_ptr = 0;
  parser.add_start_key("PostFilteringParameters");
  parser.add_parsing_key("PostFilter type", &filter_ptr);
  parser.add_stop_key("END PostFiltering Parameters");
  
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  
  shared_ptr<DiscretisedDensity<3,float> > input_image_ptr;
  PostFiltering post_filtering;
  string out_filename;
  
  if (argc!=4)
    {
      cerr<<"\nUsage: postfilter <output filename > <input header file name> <filter .par filename>\n"<<endl;
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
	DiscretisedDensity<3,float>::read_from_file(argv[2]);
    }
  else
    {
      input_image_ptr= ask_image("Image to process?");
    }
  if (argc>3)
    {
      post_filtering.parser.parse(argv[3]);
    }
  else
    {     
      cerr << "\nI'm going to ask you for the type of filter (or image processor)\n"
	"Possible values:\n";
      ImageProcessor<3,float>::list_registered_names(cerr);
      
      post_filtering.parser.ask_parameters();    
    }

  cerr << "PostFilteringParameters:\n" << post_filtering.parser.parameter_info();

  if (is_null_ptr(post_filtering.filter_ptr))
    {
      warning("postfilter: No filter set. Not writing any output.\n");
      return EXIT_FAILURE;
    }

  if (is_null_ptr(input_image_ptr))
    {
      warning("postfilter: No input image. Not writing any output.\n");
      return EXIT_FAILURE;
    }
    
  post_filtering.filter_ptr->apply(*input_image_ptr);
  
  DefaultOutputFileFormat output_file_format;
  output_file_format.write_to_file(out_filename,*input_image_ptr);

  return EXIT_SUCCESS;
}


