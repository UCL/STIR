//
// $Id: 
//

/*!
\file
\ingroup utilities
\brief Threshold normalisation data

\author Kris Thielemans 


*/
/*
    Copyright (C) 2002- 2012, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir/ProjDataInterfile.h"
#include "stir/SegmentByView.h"
#include "stir/utilities.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"

#include <numeric>
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
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
    argc>4 ? static_cast<float>(atof(argv[3])) : .01F;
  const float new_value = 
    argc>5 ? static_cast<float>(atof(argv[4])) : 1.E20F;


  shared_ptr<ProjData> proj_data_ptr
    (new ProjDataInterfile(input_proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone(),
			   output_file_name));

  Succeeded success = Succeeded::yes;
   
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
      {
	warning("Error set_segment %d\n", segment_num);   
	success = Succeeded::no;
      }

  }
  
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
