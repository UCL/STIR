
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/SegmentBySinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/Sinogram.h"
#include "stir/VectorWithOffset.h"

#include <vector>
#include <algorithm>
#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::min;
using std::max;
using std::endl;
using std::vector;
#endif

USING_NAMESPACE_STIR


/* This is a program that takes corrupted sinograms
(normally after motion correction is applied) and cuts away planes that 
have too much missing data. The amount that is cut is determied by user 
in percentage).*/



void check_for_corrupted_sinograms(int& axial_pos_to_remove_min,
				   int& axial_pos_to_remove_max,
				   const SegmentBySinogram<float> in_eff_segment,
				   const float tolerance_of_corruption)

{
  axial_pos_to_remove_min =0;
  axial_pos_to_remove_max =0;
  int number_of_corrupted_bins=0;
  int counter;
  for (int axial_pos_num =in_eff_segment.get_min_axial_pos_num();
	axial_pos_num <=(in_eff_segment.get_max_axial_pos_num()+in_eff_segment.get_min_axial_pos_num())/2; 
	axial_pos_num++)
	{	
	  counter =0;
	  Sinogram<float> sinogram = in_eff_segment.get_sinogram(axial_pos_num);
	  {
	    int total_num_bins_sino = sinogram.get_num_views()*sinogram.get_num_tangential_poss();
	    for ( int view_num = sinogram.get_min_view_num();
	    view_num <= sinogram.get_max_view_num(); view_num++)
	      for ( int tang_pos = sinogram.get_min_tangential_pos_num()+20;
	      tang_pos <= sinogram.get_max_tangential_pos_num()-20;
	      tang_pos++)
	      {
		float bin = sinogram[view_num][tang_pos];
		if (fabs(bin/0.1) < 3)
		  counter ++;
	      }
	      float percetage_corrupted = static_cast<float>(counter)/static_cast<float> (total_num_bins_sino)*100;

	      if ( percetage_corrupted >tolerance_of_corruption)
		    axial_pos_to_remove_min++;
	      
	  }
	}
	// now do the max side
	for (int axial_pos_num =(in_eff_segment.get_max_axial_pos_num()+in_eff_segment.get_min_axial_pos_num())/2+1;
	axial_pos_num <=in_eff_segment.get_max_axial_pos_num(); 
	axial_pos_num++)
	{	
	  counter =0;
	  Sinogram<float> sinogram = in_eff_segment.get_sinogram(axial_pos_num);
	  {
	    int total_num_bins_sino = sinogram.get_num_views()*sinogram.get_num_tangential_poss();
	    for ( int view_num = sinogram.get_min_view_num();
	    view_num <= sinogram.get_max_view_num(); view_num++)
	      for ( int tang_pos = sinogram.get_min_tangential_pos_num()+20;
	      tang_pos <= sinogram.get_max_tangential_pos_num()-20;
	      tang_pos++)
	      {
		float bin = sinogram[view_num][tang_pos];
		if ( fabs(bin/0.1) < 3)
		  counter ++;
	      }
	      float percetage_corrupted = static_cast<float>(counter)/static_cast<float> (total_num_bins_sino)*100;
	      if ( percetage_corrupted >tolerance_of_corruption)
		   axial_pos_to_remove_max++;
	   }
	}
	
	  
	  
}
int
main(int argc, char* argv[])
{
 
  if ( argc !=6)
  {
    cerr << "Usage:" << argv[0] << "Output filename, sinogram to modify and the corresponding efficieny file, max_number_of_segment_to_process,"
      		    << "tolerance_of_corruption(in %)" << endl;
    return EXIT_FAILURE;
  }

#if 1
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  shared_ptr<ProjData> eff_projdata_ptr = ProjData::read_from_file(argv[3]);  
 
  int max_segment_num_to_process = atoi(argv[4]);
  float tolerance_of_corruption =atoi(argv[5]);
  //int num_axial_poss_to_remove_at_min_side = atoi(argv[5]);
  //int num_axial_poss_to_remove_at_max_side = atoi(argv[6]);

  vector<int> record_of_number_of_planes_to_remove_min;
  vector<int> record_of_number_of_planes_to_remove_max;
    
  // construct new proj_data_info_ptr for output data
  ProjDataInfo * proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
  {
    
    proj_data_info_ptr->reduce_segment_range(-max_segment_num_to_process,
					     max_segment_num_to_process);
    // now set new number of axial positions
    VectorWithOffset<int> 
      new_num_axial_poss_per_segment(-max_segment_num_to_process,
				     max_segment_num_to_process);
#if 1  
    int axial_pos_to_remove_min;
    int axial_pos_to_remove_max;
    // only do segment zero and use that for all other segments
    const SegmentBySinogram<float> in_eff_segment = 
	     eff_projdata_ptr->get_segment_by_sinogram(0);
    
    check_for_corrupted_sinograms(axial_pos_to_remove_min,
	     axial_pos_to_remove_max,
	     in_eff_segment,tolerance_of_corruption);
	   
   int number_of_axial_pos_to_remove = axial_pos_to_remove_min+axial_pos_to_remove_max;
   record_of_number_of_planes_to_remove_min.push_back(axial_pos_to_remove_min);
   record_of_number_of_planes_to_remove_max.push_back(axial_pos_to_remove_max);

   new_num_axial_poss_per_segment[0] =
     in_projdata_ptr->get_num_axial_poss(0) - 
     number_of_axial_pos_to_remove;
   if (new_num_axial_poss_per_segment[0]>1)
     proj_data_info_ptr->set_num_axial_poss_per_segment(new_num_axial_poss_per_segment);

   for (int segment_num=-max_segment_num_to_process; 
	 segment_num<=max_segment_num_to_process;
	 ++segment_num)
	 {
	   if (segment_num !=0)
	   {
	     //num_axial_poss_to_remove_at_min_side+
	     //num_axial_poss_to_remove_at_max_side;
	   new_num_axial_poss_per_segment[segment_num] =
	     in_projdata_ptr->get_num_axial_poss(segment_num) - 
	     number_of_axial_pos_to_remove;
	   if (new_num_axial_poss_per_segment[segment_num]>1)
	   proj_data_info_ptr->set_num_axial_poss_per_segment(new_num_axial_poss_per_segment);
	   else
	   {
	     error("%s: there are not enough axial positions even in segment 0\n",
		argv[0]);
	   }
	   }
	   else
	     continue;
	 }
#endif
     
  }
 
  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  Succeeded succes = Succeeded::yes;
  std::vector<int>::iterator iter_record_of_number_of_planes_to_remove_min =
			record_of_number_of_planes_to_remove_min.begin();
  std::vector<int>::iterator iter_record_of_number_of_planes_to_remove_max =
			record_of_number_of_planes_to_remove_max.begin();

  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    {       
      SegmentBySinogram<float> out_segment =
	out_projdata.get_empty_segment_by_sinogram(segment_num);
      const SegmentBySinogram<float> in_segment = 
        in_projdata_ptr->get_segment_by_sinogram( segment_num);

      int num_axial_poss_to_remove_at_min_side = *iter_record_of_number_of_planes_to_remove_min;
      int num_axial_poss_to_remove_at_max_side = *iter_record_of_number_of_planes_to_remove_max;

      cerr << " Number of planes to remove on the min side is " << num_axial_poss_to_remove_at_min_side << endl;
      cerr << " Number of planes to remove on the max side is " << num_axial_poss_to_remove_at_max_side << endl;


      int ax_pos_num_out = out_segment.get_min_axial_pos_num();
      for (int ax_pos_num=(in_segment.get_min_axial_pos_num()+num_axial_poss_to_remove_at_min_side);
           ax_pos_num<=in_segment.get_max_axial_pos_num()-num_axial_poss_to_remove_at_max_side;
       ++ax_pos_num)
       {
	  cerr << ax_pos_num << "  ";
	  out_segment[ax_pos_num_out] = in_segment[ax_pos_num];
	  ax_pos_num_out++;
	
       }
       if (out_projdata.set_segment(out_segment) == Succeeded::no)
             succes = Succeeded::no;
       cerr << endl;

      
	 
    }

 
#endif


  return EXIT_SUCCESS;
}
