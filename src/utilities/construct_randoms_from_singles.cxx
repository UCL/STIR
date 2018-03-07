/*!

  \file
  \ingroup utilities

  \brief Construct randoms as a product of singles estimates

  \author Kris Thielemans

*/
/*
  Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

#include "stir/ML_norm.h"

#include "stir/ProjDataInterfile.h"


#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange2D.h"
#include "stir/display.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::fstream;
using std::string;
using std::ios;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=5)
    {
      cerr << "Usage: " << argv[0] 
           << " out_filename in_norm_filename_prefix template_projdata eff_iter_num\n";
      return EXIT_FAILURE;
    }
#if 0
  bool do_block = argc>=10?atoi(argv[9])!=0: true;
  bool do_geo   = argc>=9?atoi(argv[8])!=0: true;
  bool do_eff   = argc>=8?atoi(argv[7])!=0: true;  
#else
  bool do_eff = true;
#endif
  const int eff_iter_num = atoi(argv[4]);
  const int iter_num = 1;//atoi(argv[5]);
  //const bool apply_or_undo = atoi(argv[4])!=0;
  shared_ptr<ProjData> template_projdata_ptr = ProjData::read_from_file(argv[3]);
  const string in_filename_prefix = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];

  ProjDataInterfile 
    proj_data(template_projdata_ptr->get_exam_info_sptr(),
              template_projdata_ptr->get_proj_data_info_ptr()->create_shared_clone(), 
	      output_file_name);

  const int num_rings = 
    template_projdata_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    template_projdata_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
#if 0
  const int num_tangential_crystals_per_block = 8;
  const int num_tangential_blocks = num_detectors_per_ring/num_tangential_crystals_per_block;
  const int num_axial_crystals_per_block = num_rings/2;
  warning("TODO num_axial_crystals_per_block == num_rings/2\n");
  const int num_axial_blocks = num_rings/num_axial_crystals_per_block;

  BlockData3D norm_block_data(num_axial_blocks, num_tangential_blocks,
                              num_axial_blocks-1, num_tangential_blocks-1);
#endif
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

  {

    // efficiencies
    if (do_eff)
      {
	char *in_filename = new char[in_filename_prefix.size() + 30];
	sprintf(in_filename, "%s_%s_%d_%d.out", 
		in_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	ifstream in(in_filename);
	in >> efficiencies;
	if (!in)
	  {
	    warning("Error reading %s, using all 1s instead\n", in_filename);
	    do_eff = false;
	  }
	delete[] in_filename;
      }
#if 0
    // block norm
    if (do_block)
      {
	{
	  char *in_filename = new char[in_filename_prefix.size() + 30];
	  sprintf(in_filename, "%s_%s_%d.out", 
		  in_filename_prefix.c_str(), "block",  iter_num);
	  ifstream in(in_filename);
	  in >> norm_block_data;
	  if (!in)
	    {
	      warning("Error reading %s, using all 1s instead\n", in_filename);
	      do_block = false;
	    }
	  delete[] in_filename;
	}
      }
#endif
  }

  {
    const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr = 
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>
      (proj_data.get_proj_data_info_ptr());
    if (proj_data_info_ptr == 0)
      {
	error("Can only process not arc-corrected data\n");
      }
    const int max_ring_diff = 
      proj_data_info_ptr->get_max_ring_difference
      (proj_data_info_ptr->get_max_segment_num());

    const int mashing_factor = 
      proj_data_info_ptr->get_view_mashing_factor();

    shared_ptr<Scanner> scanner_sptr(new Scanner(*proj_data_info_ptr->get_scanner_ptr()));
    unique_ptr<ProjDataInfo> uncompressed_proj_data_info_uptr
    (ProjDataInfo::construct_proj_data_info(scanner_sptr,
      /*span=*/1, max_ring_diff,
      /*num_views=*/num_detectors_per_ring / 2,
      scanner_sptr->get_max_num_non_arccorrected_bins(),
      /*arccorrection=*/false));
    const ProjDataInfoCylindricalNoArcCorr * const
      uncompressed_proj_data_info_ptr =
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>
      (uncompressed_proj_data_info_uptr.get());

    
    Bin bin;
    Bin uncompressed_bin;

    for (bin.segment_num() = proj_data.get_min_segment_num(); 
	 bin.segment_num() <= proj_data.get_max_segment_num();  
	 ++ bin.segment_num())
      {	
    
	for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
	     bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
	     ++bin.axial_pos_num())
	  {
	    Sinogram<float> sinogram =
	      proj_data_info_ptr->get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());
	    const float out_m = proj_data_info_ptr->get_m(bin);
	    const int in_min_segment_num =
	      proj_data_info_ptr->get_min_ring_difference(bin.segment_num());
	    const int in_max_segment_num =
	      proj_data_info_ptr->get_max_ring_difference(bin.segment_num());

	    // now loop over uncompressed detector-pairs
	  
	    {  
	      for (uncompressed_bin.segment_num() = in_min_segment_num; 
		   uncompressed_bin.segment_num() <= in_max_segment_num;
		   ++uncompressed_bin.segment_num())
		for (uncompressed_bin.axial_pos_num() = uncompressed_proj_data_info_ptr->get_min_axial_pos_num(uncompressed_bin.segment_num()); 
		     uncompressed_bin.axial_pos_num()  <= uncompressed_proj_data_info_ptr->get_max_axial_pos_num(uncompressed_bin.segment_num());
		     ++uncompressed_bin.axial_pos_num() )
		  {
		    const float in_m = uncompressed_proj_data_info_ptr->get_m(uncompressed_bin);
		    if (fabs(out_m - in_m) > 1E-4)
		      continue;
		

		    // views etc
		    if (proj_data.get_min_view_num()!=0)
		      error("Can only handle min_view_num==0\n");
		    for (bin.view_num() = proj_data.get_min_view_num(); 
			 bin.view_num() <= proj_data.get_max_view_num();
			 ++ bin.view_num())
		      {

			for (bin.tangential_pos_num() = proj_data_info_ptr->get_min_tangential_pos_num();
			     bin.tangential_pos_num() <= proj_data_info_ptr->get_max_tangential_pos_num();
			     ++bin.tangential_pos_num())
			  {
			    uncompressed_bin.tangential_pos_num() =
			      bin.tangential_pos_num();
			    for (uncompressed_bin.view_num() = bin.view_num()*mashing_factor;
				 uncompressed_bin.view_num() < (bin.view_num()+1)*mashing_factor;
				 ++ uncompressed_bin.view_num())
			      {

				int ra = 0, a = 0;
				int rb = 0, b = 0;
			      
				uncompressed_proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, 
										      uncompressed_bin);

				/*(*segment_ptr)[bin.axial_pos_num()]*/
				sinogram[bin.view_num()][bin.tangential_pos_num()] +=
				  efficiencies[ra][a]*efficiencies[rb][b%num_detectors_per_ring];
			      }
			  }
		      }
		  
		  
		  }
	    }
	    proj_data.set_sinogram(sinogram);
	  }

      }
  }


  return EXIT_SUCCESS;
}
