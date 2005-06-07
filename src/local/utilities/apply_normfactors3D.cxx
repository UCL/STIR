//
// $Id$
//
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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

  \brief Apply normalisation factors to projection data

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
  if (argc<7 || argc>12)
    {
      std::cerr << "Usage: " << argv[0] 
           << " out_filename in_norm_filename_prefix measured_data apply_or_undo iter_num eff_iter_num [do_eff [ do_geo [ do_block [do_display]]]]\n"
	   << "apply_or_undo is 1 (multiply) or 0 (divide)\n"
	   << "do_eff, do_geo, do_block are 1 or 0 and all default to 1\n"	   
      	   << "do_display is 1 or 0 (defaults to 0)\n";
      return EXIT_FAILURE;
    }

  const bool do_display = argc>=11?atoi(argv[10])!=0 : false;
  const bool do_block = argc>=10?atoi(argv[9])!=0: true;
  const bool do_geo   = argc>=9?atoi(argv[8])!=0: true;
  const bool do_eff   = argc>=8?atoi(argv[7])!=0: true;  

  if (do_geo)
    error("Cannot do geometric factors in 3D yet");
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

  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_rings();
  const int num_detectors_per_ring = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_detectors_per_ring();
  const int num_transaxial_blocks =
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_blocks();
  const int num_axial_blocks =
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_axial_blocks();

  BlockData3D norm_block_data(num_axial_blocks, num_transaxial_blocks,
                              num_axial_blocks-1, num_transaxial_blocks-1);
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

    {

      // efficiencies
      if (do_eff)
  	{
	  char *in_filename = new char[in_filename_prefix.size() + 30];
	  sprintf(in_filename, "%s_%s_%d_%d.out", 
		  in_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	  std::ifstream in(in_filename);
	  in >> efficiencies;
	    if (!in)
	      {
		warning("Error reading %s, using all 1s instead\n", in_filename);
		do_eff = false;
	      }
	  delete in_filename;
	}
      	// block norm
      if (do_block)
	{
	  {
	    char *in_filename = new char[in_filename_prefix.size() + 30];
	    sprintf(in_filename, "%s_%s_%d.out", 
		    in_filename_prefix.c_str(), "block",  iter_num);
	    std::ifstream in(in_filename);
	    in >> norm_block_data;
	    if (!in)
	      {
		warning("Error reading %s, using all 1s instead\n", in_filename);
	        do_block = false;
	      }
	    delete in_filename;
	  }
	}

      {
        FanProjData fan_data;
	make_fan_data(fan_data, *measured_data);
	if (do_eff)
          apply_efficiencies(fan_data, efficiencies, apply_or_undo);
       	//if (do_geo)
	//  apply_geo_norm(fan_data, norm_geo_data, apply_or_undo);
	if (do_block)
	  apply_block_norm(fan_data, norm_block_data, apply_or_undo);

	if (do_display)
	  display(fan_data, "input*norm");
	set_fan_data(*out_proj_data_ptr, fan_data);
      }
    }

  return EXIT_SUCCESS;
}
