//
// $Id: 
//

/*!
\file
\ingroup utilities
\brief Zero projection data when corresponding normalisation factors are too high

\author Kris Thielemans 

$Date$
$Revision$ 

*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInterfile.h"
#include "stir/SegmentByView.h"
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

  
  if (argc<4 || argc>5)
  {
    cerr << " USAGE: " << argv[0] << " output_projdata_filename input_projdata_filename norm_projdata_filename [max_threshold_in_norm]\n"
	 << "   max_threshold_in_norm defaults to 1E19"
	 << endl;
    exit(EXIT_FAILURE);
  }


  const string output_file_name = argv[1];
  shared_ptr<ProjData> input_proj_data_ptr = 
    ProjData::read_from_file(argv[2]); 
  shared_ptr<ProjData> norm_proj_data_ptr = 
    ProjData::read_from_file(argv[3]); 
  const float max_threshold_in_norm = 
    argc>5 ? static_cast<float>(atof(argv[4])) : 1.E19F;

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
    const SegmentByView<float> norm_segment  = 
      norm_proj_data_ptr->get_segment_by_view(segment_num);

    SegmentByView<float>::full_iterator iter = segment.begin_all();
    SegmentByView<float>::const_full_iterator norm_iter = norm_segment.begin_all();
    for (;
	 iter != segment.end_all();
	 ++iter, ++norm_iter)
      {
	if (*norm_iter > max_threshold_in_norm)
	  *iter = 0;
      }
    if (!(proj_data_ptr->set_segment(segment) == Succeeded::yes))
      {
	warning("Error writing segment %d\n", segment_num);
	success = Succeeded::no;
      }
  }
  
  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
