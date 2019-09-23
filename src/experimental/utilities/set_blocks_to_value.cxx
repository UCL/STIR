//
//
/*!

  \file
  \ingroup utilities
  \brief A utility to set data corresponding to a certain detector block to a desired value

  \author Kris Thielemans

*/
/*
    Copyright (C) 2002- 2004, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/Viewgram.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInterfile.h"
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::vector;
using std::string;
using std::cerr;
using std::fstream;
using std::sort;
using std::unique;
using std::min;
using std::max;
#endif
START_NAMESPACE_STIR



void do_block(vector<Bin>& list_of_bins_in_block, 
              const int axial_block_num, const int tangential_block_num,
              const ProjDataInfoCylindricalNoArcCorr& proj_data_info,
              const int axial_num_crystals_in_block, const int tangential_num_crystals_in_block)
{
  Bin bin;
  const int num_rings = 
    proj_data_info.get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  
  const int tang_det_offset = tangential_block_num*tangential_num_crystals_in_block;
  const int ax_det_offset = axial_block_num*axial_num_crystals_in_block;
  const int max_ring_diff = 
    proj_data_info.get_max_ring_difference(proj_data_info.get_max_segment_num());
  const int min_ring_diff = 
    proj_data_info.get_min_ring_difference(proj_data_info.get_min_segment_num());
  for (int ax_crystal=0; ax_crystal<axial_num_crystals_in_block; ++ax_crystal)
    for (int tang_crystal=0; tang_crystal<tangential_num_crystals_in_block; ++tang_crystal)
    {
      const int det = tang_crystal + tang_det_offset;
      const int ring = ax_crystal + ax_det_offset;
      {
        for (int other_det=0; other_det<num_detectors_per_ring; ++other_det)
          for (int other_ring=max(0,ring+min_ring_diff); other_ring<=min(num_rings-1,ring+max_ring_diff); ++other_ring)
          {
            // first check for impossible coincidence (because it cannot be handled by get_bin_for_det_pair)
            if (det == other_det)
              continue;
            Succeeded success =
              proj_data_info.get_bin_for_det_pair(bin, 
              det, ring,
              other_det, other_ring);
            if (success == Succeeded::yes)
              list_of_bins_in_block.push_back(bin);
          }
      }
    }
}

bool 
bin_coordinates_by_view_less(const Bin& b1, const Bin& b2)
{
  return b1.segment_num()<b2.segment_num() ||
      (b1.segment_num()==b2.segment_num()&& 
       (b1.view_num()<b2.view_num() || 
        (b1.view_num()==b2.view_num() && 
         (b1.axial_pos_num()<b2.axial_pos_num() ||
          (b1.axial_pos_num()==b2.axial_pos_num() &&
           b1.tangential_pos_num()<b2.tangential_pos_num())))));
}

void sort_and_make_unique(vector<Bin>& list_of_bins)
{
  // cannot use vector::sort as VC does not support member templates
  sort(list_of_bins.begin(), list_of_bins.end(),bin_coordinates_by_view_less);
  //list_of_bins.unique();
  unique(list_of_bins.begin(), list_of_bins.end());
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc < 3 || argc > 5)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name [value [max_in_segment_num_to_process ]]\n"
	   << "value defaults to 0\n"
           << "max_in_segment_num_to_process defaults to all segments\n"
	   << "Will ask for which blocks to set to the value, will then set "
	   << "ANY bin that has a contribution of those blocks to that value.\n"
	   << "This might not be what you want for spanned/mashed data.\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  const float value = argc <=3 ? 0.F: static_cast<float>(atof(argv[3]));
  const int max_segment_num_to_process = argc <=4 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[4]);

  ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr * >
      (in_projdata_ptr->get_proj_data_info_ptr()->clone());
  if (proj_data_info_ptr == NULL)
  {
    cerr << argv[0] << " can only work on not-arccorrected data\n";
    exit(EXIT_FAILURE);
  }
  proj_data_info_ptr->reduce_segment_range(-max_segment_num_to_process,max_segment_num_to_process);

  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  const int num_rings = 
    proj_data_info_ptr->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring();
  const int axial_num_crystals_in_block = 
    ask_num("Crystals in 1 block axially",1,num_rings,8);
  const int tangential_num_crystals_in_block= 
    ask_num("Crystals in 1 block tangentially",1,num_detectors_per_ring,8);

  vector<Bin> list_of_bins;
  do
  {
     const int axial_block_num = 
       ask_num("Block number axially",
	       0,num_rings/axial_num_crystals_in_block-1, 0);
     const int tangential_block_num =
       ask_num("Block number tangentially",
	       0,num_detectors_per_ring/tangential_num_crystals_in_block-1, 0);
     do_block(list_of_bins, 
              axial_block_num, tangential_block_num,
              *proj_data_info_ptr,
              axial_num_crystals_in_block, tangential_num_crystals_in_block);
  }
  while (ask("One more",false));


  sort_and_make_unique(list_of_bins);

  std::vector<Bin>::const_iterator bin_iter = list_of_bins.begin();
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)
    for (int view_num = in_projdata_ptr->get_min_view_num();
         view_num <= in_projdata_ptr->get_max_view_num();
         ++view_num)
    {       
      Viewgram<float> viewgram =
	out_projdata.get_empty_viewgram(view_num, segment_num);
      viewgram +=  in_projdata_ptr->get_viewgram(view_num, segment_num);
      const int max_ax_pos_num = viewgram.get_max_axial_pos_num();
      const int min_ax_pos_num = viewgram.get_min_axial_pos_num();
      const int max_tang_pos_num = viewgram.get_max_tangential_pos_num();
      const int min_tang_pos_num = viewgram.get_min_tangential_pos_num();
      for (; bin_iter != list_of_bins.end(); ++bin_iter)
      {
        if (segment_num != bin_iter->segment_num() ||
          view_num != bin_iter->view_num())
          break;
	const int ax_pos_num = bin_iter->axial_pos_num();
	const int tang_pos_num = bin_iter->tangential_pos_num();
	if (ax_pos_num <=max_ax_pos_num &&
	    ax_pos_num >=min_ax_pos_num &&
	    tang_pos_num <=max_tang_pos_num &&
	    tang_pos_num >=min_tang_pos_num)
	  viewgram[ax_pos_num][tang_pos_num] = value;
      }      
      out_projdata.set_viewgram(viewgram);
    }

  return EXIT_SUCCESS;
}
