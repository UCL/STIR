//
// $Id$
//

/*!
\file
\ingroup utilities
\brief add sinogram data

\warning There are no checks on compatibility of the projection data you are adding

\author Sanida Mustafovic 
\author Kris Thielemans 

$Date$
$Revision$ 
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"


#include <fstream> 
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
#endif



USING_NAMESPACE_STIR

int 
main(int argc, char **argv)

{
  if(argc<4)
  {
    cerr<< "Usage: " << argv[0] << " out_projdata projdata1 projdata2 [projdata3...]\n";
    exit(EXIT_FAILURE);
  }
  const char * const program_name = argv[0];

  --argc;
  ++argv;
  const string output_file_name = *argv;

  const int num_files = --argc;
  ++argv;
  vector< shared_ptr<ProjData> > all_proj_data(num_files);

  // read all projection data headers
  for (int i=0; i<num_files; ++i)
    all_proj_data[i] =  ProjData::read_from_file(argv[i]); 

  const ProjDataInfo * proj_data_info_ptr = 
    (*all_proj_data[0]).get_proj_data_info_ptr();

  // opening output file
  shared_ptr<iostream> sino_stream = new fstream (output_file_name.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("%s: error opening output file %s\n",
	  program_name, output_file_name.c_str());
  }

  shared_ptr<ProjDataFromStream> proj_data_ptr =
    new ProjDataFromStream(proj_data_info_ptr->clone(),sino_stream);
  write_basic_interfile_PDFS_header(output_file_name, *proj_data_ptr);
   

  // do reading/writing in a loop over segments
  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <=proj_data_info_ptr->get_max_segment_num();
       segment_num++)
  {   
    SegmentByView<float> segment_by_view = 
      (*all_proj_data[0]).get_segment_by_view(segment_num);
    for (int i=1; i<num_files; ++i)
       segment_by_view += 
	 (*all_proj_data[i]).get_segment_by_view(segment_num);

    
    if (!(proj_data_ptr->set_segment(segment_by_view) == Succeeded::yes))
      warning("Error set_segment %d\n", segment_num);   
  }
  
  return EXIT_SUCCESS;
}
