//
// $Id$
//

/*!
  \file 
  \ingroup utilities

  \brief  This programme performs filtering on image data
 
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic (conversion to ImageProcessor)
  \author PARAPET project
  
  \date $Date$
  \version $Revision$

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


  
  \warning It only supports VoxelsOnCartesianGrid type of images.
*/


#include "interfile.h"
#include "utilities.h"
#include "KeyParser.h"
#include "DiscretisedDensity.h"
#include "VoxelsOnCartesianGrid.h"
#include "tomo/ImageProcessor.h"

#include <iostream> 
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
#endif


START_NAMESPACE_TOMO

VoxelsOnCartesianGrid<float>* ask_interfile_image(char *input_query);




/***************** Miscellaneous Functions  *******/



VoxelsOnCartesianGrid<float>* ask_interfile_image(char *input_query){
  
  
  char filename[max_filename_length];
  ask_filename_with_extension(filename, 
				input_query,
				"");
  
  return read_interfile_image(filename);
  
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
  
  VoxelsOnCartesianGrid<float> input_image;
  PostFiltering post_filtering;
  string out_filename;
  
  if (argc==4)
  {
    out_filename = argv[1];
    input_image = 
      * dynamic_cast<VoxelsOnCartesianGrid<float> *>(
      DiscretisedDensity<3,float>::read_from_file(argv[2]));

    post_filtering.parser.parse(argv[3]);
  }
  else
  {
    cerr<<endl<<"Usage: postfilter <output filename > <input header file name> <filter .par filename>"<<endl<<endl;
    input_image= *ask_interfile_image("Image to process?");
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
    
  post_filtering.filter_ptr->build_and_filter(input_image);
  
  
  write_basic_interfile(out_filename.c_str(),input_image);

  
  return EXIT_SUCCESS;
}


