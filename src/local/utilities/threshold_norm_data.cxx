//
// $Id: 
//

/*!
\file
\ingroup utilities
\brief Threshold normalisation data

\author Kris Thielemans 

$Date$
$Revision$ 

*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/shared_ptr.h"

#include <numeric>
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

  
  
  if (argc<3 || argc>5)
  {
    cerr << " USAGE: threshold_norm_data output_projdata_filename input_projdata_filename [min_threshold  [new_value]]\n"
	 << "   new_value defaults to 1E20\n"
	 << "   min_threshold defaults to .01"
	 << endl;
    exit(EXIT_FAILURE);
  }

  const string output_file_name = argv[1];
  shared_ptr<ProjData> input_proj_data_ptr = 
    ProjData::read_from_file(argv[2]); 
  const float min_threshold = 
    argc>4 ? atof(argv[3]) : .01F;
  const float new_value = 
    argc>5 ? atof(argv[4]) : 1.E20F;

  shared_ptr<iostream> sino_stream = 
    new fstream (output_file_name.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("threshold_norm_data: error opening file %s\n",output_file_name.c_str());
  }

  shared_ptr<ProjDataFromStream> proj_data_ptr =
    new ProjDataFromStream(input_proj_data_ptr->get_proj_data_info_ptr()->clone(),
			   sino_stream);
  write_basic_interfile_PDFS_header(output_file_name, *proj_data_ptr);
   
  for (int segment_num = input_proj_data_ptr->get_min_segment_num();
       segment_num <=input_proj_data_ptr->get_max_segment_num();
       ++segment_num)
  {   
    SegmentByView<float> segment  = 
      input_proj_data_ptr->get_segment_by_view(segment_num);

    for (SegmentByView<float>::full_iterator iter = segment.begin_all();
	 iter != segment.end_all();
	 ++iter)
      {
	if (*iter < min_threshold)
	  *iter = new_value;
      }
    if (!(proj_data_ptr->set_segment(segment) == Succeeded::yes))
      warning("Error set_segment %d\n", segment_num);   
  }
  
  

  return EXIT_SUCCESS;
}
