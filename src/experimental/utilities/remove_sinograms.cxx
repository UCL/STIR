//
//
/*!

  \file
  \ingroup utilities
  \brief A utility to remove (or add) some sinograms at the end of each segment

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
using std::endl;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if (argc < 4 || argc > 5)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name num_axial_poss_to_remove_at_each_side [max_in_segment_num_to_process ]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n"
	   << "If num_axial_poss_to_remove_at_each_side is negative, sinograms will \n"
	   << "be added (they will be set to 0)\n";
      exit(EXIT_FAILURE);
    }
 
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  const int num_axial_poss_to_remove_at_each_side =  atoi(argv[3]);
  int max_segment_num_to_process = argc <=4 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[4]);
 
  // construct new proj_data_info_ptr for output data
  ProjDataInfo * proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
  {
    // first check that max_segment_num_to_process is such that in the new
    // number of axial positions is positive. If not, decrease it
    while (max_segment_num_to_process >=0 &&
	   in_projdata_ptr->get_num_axial_poss(max_segment_num_to_process) - 
	   2*num_axial_poss_to_remove_at_each_side<=0)
      --max_segment_num_to_process;

    if (max_segment_num_to_process<0)
      {
	error("%s: there are not enough axial positions even in segment 0\n",
		argv[0]);
      }
   

    proj_data_info_ptr->reduce_segment_range(-max_segment_num_to_process,
					     max_segment_num_to_process);
    // now set new number of axial positions
    VectorWithOffset<int> 
      new_num_axial_poss_per_segment(-max_segment_num_to_process,
				     max_segment_num_to_process);
    for (int segment_num=-max_segment_num_to_process; 
	 segment_num<=max_segment_num_to_process;
	 ++segment_num)
      new_num_axial_poss_per_segment[segment_num] =
	in_projdata_ptr->get_num_axial_poss(segment_num) - 
	2*num_axial_poss_to_remove_at_each_side;
	

    proj_data_info_ptr->set_num_axial_poss_per_segment(new_num_axial_poss_per_segment);
     
  }
 
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

      // 17/02/2002 SM corrected such that sinograms from both sides are removed not only from one side
      for (int ax_pos_num=(in_segment.get_min_axial_pos_num()+num_axial_poss_to_remove_at_each_side);
           ax_pos_num<=in_segment.get_max_axial_pos_num()-num_axial_poss_to_remove_at_each_side;
       ++ax_pos_num)
       {
	 cerr << ax_pos_num << "  ";
        out_segment[ax_pos_num-num_axial_poss_to_remove_at_each_side] = in_segment[ax_pos_num];
	
       }
       if (out_projdata.set_segment(out_segment) == Succeeded::no)
             succes = Succeeded::no;
       cerr << endl;

      
	 
    }

    return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
