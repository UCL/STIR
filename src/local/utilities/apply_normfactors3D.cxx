//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief Apply normalisation factors to projection data

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/ML_norm.h"

#include "stir/ProjDataFromStream.h"
#include "stir/interfile.h"

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
  if (argc<7 || argc>12)
    {
      cerr << "Usage: " << argv[0] 
           << " out_filename in_norm_filename_prefix measured_data apply_or_undo iter_num eff_iter_num [do_eff [ do_geo [ do_block [do_display]]]]\n"
	   << "apply_or_undo is 1 (multiply) or 0 (divide)\n"
	   << "do_eff, do_geo, do_block are 1 or 0 and all default to 1\n"	   
      	   << "do_display is 1 or 0 (defaults to 0)\n";
      return EXIT_FAILURE;
    }

  const bool do_display = argc>=11?atoi(argv[10])!=0 : false;
  bool do_block = argc>=10?atoi(argv[9])!=0: true;
  bool do_geo   = argc>=9?atoi(argv[8])!=0: true;
  bool do_eff   = argc>=8?atoi(argv[7])!=0: true;  

  const int eff_iter_num = atoi(argv[6]);
  const int iter_num = atoi(argv[5]);
  const bool apply_or_undo = atoi(argv[4])!=0;
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[3]);
  const string in_filename_prefix = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];

  shared_ptr<ProjDataFromStream> out_proj_data_ptr;
  {
    const ProjDataInfo * proj_data_info_ptr = 
      measured_data->get_proj_data_info_ptr();

    // opening output file
    shared_ptr<iostream> sino_stream = 
      new fstream (output_file_name.c_str(), ios::out|ios::binary);
    if (!sino_stream->good())
      {
	error("%s: error opening output file %s\n",
	      program_name.c_str(), output_file_name.c_str());
      }

    out_proj_data_ptr =
      new ProjDataFromStream(proj_data_info_ptr->clone(),sino_stream);
    write_basic_interfile_PDFS_header(output_file_name, *out_proj_data_ptr);
  }

  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_tangential_crystals_per_block = 8;
  const int num_tangential_blocks = num_detectors_per_ring/num_tangential_crystals_per_block;
  const int num_axial_crystals_per_block = num_rings/2;
  warning("TODO num_axial_crystals_per_block == num_rings/2\n");
  const int num_axial_blocks = num_rings/num_axial_crystals_per_block;

  BlockData3D norm_block_data(num_axial_blocks, num_tangential_blocks,
                              num_axial_blocks-1, num_tangential_blocks-1);
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
	  delete in_filename;
	}
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
