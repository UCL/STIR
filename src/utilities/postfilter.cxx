//
// $Id$
//

/*!
  \file 
  \ingroup utilities

  \brief  This programme performs filtering on image data
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project
  
  \date $Date$
  \version $Revision$

  This programme enables calling any ImageProcessor object on input data, 
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

#include "interfile.h"
#include "utilities.h"
#include "KeyParser.h"
#include "DiscretisedDensity.h"
#include "tomo/ImageProcessor.h"

#include <iostream> 

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

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
  ImageProcessor<3,float>* filter_ptr;
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


END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO

int
main(int argc, char *argv[])
{
  
  DiscretisedDensity<3,float> *input_image_ptr;
  PostFiltering post_filtering;
  string out_filename;
  
  if (argc==4)
  {
    out_filename = argv[1];
    input_image_ptr = 
      DiscretisedDensity<3,float>::read_from_file(argv[2]);

    post_filtering.parser.parse(argv[3]);
  }
  else
  {
    cerr<<endl<<"Usage: postfilter <output filename > <input header file name> <filter .par filename>"<<endl<<endl;
    input_image_ptr= ask_image("Image to process?");
    char outfile[max_filename_length];
    ask_filename_with_extension(outfile,
      "Output to which file: ", "");
    out_filename = outfile;
  
    post_filtering.parser.ask_parameters();    
  }

  cerr << "PostFilteringParameters:\n" << post_filtering.parser.parameter_info();

  if (post_filtering.filter_ptr == 0)
    {
      error("postfilter: No filter set. Not writing any output.\n");
    }

  if (input_image_ptr == 0)
    {
      error("postfilter: No input image. Not writing any output.\n");
    }
    
  post_filtering.filter_ptr->build_and_filter(*input_image_ptr);
  
  
  write_basic_interfile(out_filename.c_str(),*input_image_ptr);

  delete input_image_ptr;
  
  return EXIT_SUCCESS;
}


