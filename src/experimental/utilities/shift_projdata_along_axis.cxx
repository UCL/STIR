//
//
/*!

  \file
  \ingroup utilities
  \brief A utility to shift projection data along the axial direction

  This can be used as a crude way for motion correction, when the motion is only in 
  z-direction.

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2003, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/SegmentBySinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include <string>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::min;
using std::max;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if (argc < 4 || argc > 5)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name number_of_axial_poss_to_shift [max_in_segment_num_to_process ]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  const int number_of_axial_poss_to_shift =  atoi(argv[3]);
  const int max_segment_num_to_process = argc <=4 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[4]);

  ProjDataInfo * proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
  proj_data_info_ptr->reduce_segment_range(-max_segment_num_to_process,max_segment_num_to_process);

  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  Succeeded succes = Succeeded::yes;
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    {       
      SegmentBySinogram<float> out_segment =
	out_projdata.get_empty_segment_by_sinogram(segment_num);
      const SegmentBySinogram<float> in_segment = 
        in_projdata_ptr->get_segment_by_sinogram( segment_num);
      const int max_ax_pos_num = out_segment.get_max_axial_pos_num();
      const int min_ax_pos_num = out_segment.get_min_axial_pos_num();
      for (int ax_pos_num=max(min_ax_pos_num, min_ax_pos_num-number_of_axial_poss_to_shift);
           ax_pos_num<=min(max_ax_pos_num, max_ax_pos_num-number_of_axial_poss_to_shift);
           ++ax_pos_num)
        out_segment[ax_pos_num] = in_segment[ax_pos_num+number_of_axial_poss_to_shift];
      if (out_projdata.set_segment(out_segment) == Succeeded::no)
             succes = Succeeded::no;
    }

    return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
