//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup utilities

  \brief Apply normalisation factors using an ML approach
  \todo should be replaced by using stir::BinNormalisationFromML2D

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "local/stir/ML_norm.h"

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


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc<7 || argc>10)
    {
      std::cerr << "Usage: " << argv[0] 
	   << " out_filename in_norm_filename_prefix measured_data apply_or_undo iter_num eff_iter_num [do_eff [ do_geo [ do_block]]]\n"
	   << "apply_or_undo is 1 (multiply) or 0 (divide)\n"
	   << "do_eff, do_geo, do_block are 1 or 0 and all default to 1\n";
      return EXIT_FAILURE;
    }

  const bool do_block = argc>=10?atoi(argv[9])!=0: true;
  const bool do_geo   = argc>=9?atoi(argv[8])!=0: true;
  const bool do_eff   = argc>=8?atoi(argv[7])!=0: true;
  const int eff_iter_num = atoi(argv[6]);
  const int iter_num = atoi(argv[5]);
  const bool apply_or_undo = atoi(argv[4])!=0;
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[3]);
  const std::string in_filename_prefix = argv[2];
  const std::string output_file_name = argv[1];
  const std::string program_name = argv[0];

  shared_ptr<ProjData> out_proj_data_ptr=
      new ProjDataInterfile(measured_data->get_proj_data_info_ptr()->clone(),
			    output_file_name);

  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_crystals_per_block();
  const int num_blocks = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_blocks();

  const int segment_num = 0;
  Array<1,float> efficiencies(num_detectors);
  assert(num_crystals_per_block%2 == 0);
  GeoData norm_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  BlockData norm_block_data(IndexRange2D(num_blocks, num_blocks));
  DetPairData det_pair_data;

  for (int ax_pos_num = measured_data->get_min_axial_pos_num(segment_num);
       ax_pos_num <= measured_data->get_max_axial_pos_num(segment_num);
       ++ax_pos_num)
    {

      // efficiencies
      if (do_eff)
	{
	  char *in_filename = new char[in_filename_prefix.size() + 30];
	  sprintf(in_filename, "%s_%s_%d_%d_%d.out", 
		  in_filename_prefix.c_str(), "eff", ax_pos_num, iter_num, eff_iter_num);
	  std::ifstream in(in_filename);
	  in >> efficiencies;
	    if (!in)
	      {
		warning("Error reading %s, using all 1s instead\n", in_filename);
		efficiencies = Array<1,float>(num_detectors);
		efficiencies.fill(1);
	      }

	  delete[] in_filename;
	}
	// geo norm
      if (do_geo)
	{
	  {
	    char *in_filename = new char[in_filename_prefix.size() + 30];
	    sprintf(in_filename, "%s_%s_%d_%d.out", 
		    in_filename_prefix.c_str(), "geo", ax_pos_num, iter_num);
	    std::ifstream in(in_filename);
	    in >> norm_geo_data;
	    if (!in)
	      {
		warning("Error reading %s, using all 1s instead\n", in_filename);
		norm_geo_data= GeoData(IndexRange2D(num_crystals_per_block/2, num_detectors));
		norm_geo_data.fill(1);
	      }
	    delete[] in_filename;
	  }
	}
	// block norm
      if (do_block)
	{
	  {
	    char *in_filename = new char[in_filename_prefix.size() + 30];
	    sprintf(in_filename, "%s_%s_%d_%d.out", 
		    in_filename_prefix.c_str(), "block", ax_pos_num, iter_num);
	    std::ifstream in(in_filename);
	    in >> norm_block_data;
	    if (!in)
	      {
		warning("Error reading %s, using all 1s instead\n", in_filename);
		norm_block_data = BlockData(IndexRange2D(num_blocks, num_blocks));
		norm_block_data.fill(1);
	      }
	    delete[] in_filename;
	  }
	}
      {
	make_det_pair_data(det_pair_data, *measured_data, segment_num, ax_pos_num);
	if (do_eff)
	  apply_efficiencies(det_pair_data, efficiencies, apply_or_undo);
	if (do_geo)
	  apply_geo_norm(det_pair_data, norm_geo_data, apply_or_undo);
	if (do_block)
	  apply_block_norm(det_pair_data, norm_block_data, apply_or_undo);
	set_det_pair_data(*out_proj_data_ptr,
			  det_pair_data,
			  segment_num,
			  ax_pos_num);
      }
    }

  return EXIT_SUCCESS;
}
