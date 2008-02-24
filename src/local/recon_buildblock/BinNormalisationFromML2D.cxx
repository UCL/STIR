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
  \ingroup recon_buildblock

  \brief Implementation for class stir::BinNormalisationFromML2D

  \author Kris Thielemans
  $Date$
  $Revision$
*/


#include "local/stir/recon_buildblock/BinNormalisationFromML2D.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "local/stir/ML_norm.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

const char * const 
BinNormalisationFromML2D::registered_name = "From ML2D"; 


void 
BinNormalisationFromML2D::set_defaults()
{
  normalisation_filename_prefix="";
  do_block = true;
  do_geo = true;
  do_eff = true;
  eff_iter_num=0;
  iter_num=0;
}

void 
BinNormalisationFromML2D::
initialise_keymap()
{
  parser.add_start_key("Bin Normalisation From ML2D");
  parser.add_key("normalisation_filename_prefix", &normalisation_filename_prefix);
  parser.add_key("use block factors", &do_block);
  parser.add_key("use geometric factors", &do_geo);
  parser.add_key("use crystal_efficiencies", &do_eff);
  parser.add_key("efficiency iteration number", &eff_iter_num);
  parser.add_key("iteration number", &iter_num);
  parser.add_stop_key("End Bin Normalisation From ML2D");
}

bool 
BinNormalisationFromML2D::
post_processing()
{
  return false;
}

BinNormalisationFromML2D::
BinNormalisationFromML2D()
{
  set_defaults();
}

Succeeded 
BinNormalisationFromML2D::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  norm_factors_ptr= new ProjDataInMemory(proj_data_info_ptr, false /* i.e. do not initialise */);
  const int num_detectors = 
    proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 
    proj_data_info_ptr->get_scanner_ptr()->
    get_num_transaxial_crystals_per_block();
  const int num_blocks = 
    proj_data_info_ptr->get_scanner_ptr()->
    get_num_transaxial_blocks();

  const int segment_num = 0;
  Array<1,float> efficiencies(num_detectors);
  assert(num_crystals_per_block%2 == 0);
  GeoData norm_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  BlockData norm_block_data(IndexRange2D(num_blocks, num_blocks));
  DetPairData det_pair_data;

  for (int ax_pos_num = proj_data_info_ptr->get_min_axial_pos_num(segment_num);
       ax_pos_num <= proj_data_info_ptr->get_max_axial_pos_num(segment_num);
       ++ax_pos_num)
    {

      // efficiencies
      if (do_eff)
	{
	  char *normalisation_filename = new char[normalisation_filename_prefix.size() + 30];
	  sprintf(normalisation_filename, "%s_%s_%d_%d_%d.out", 
		  normalisation_filename_prefix.c_str(), "eff", ax_pos_num, iter_num, eff_iter_num);
	  ifstream in(normalisation_filename);
	  in >> efficiencies;
	    if (!in)
	      {
		warning("BinNormalisationFromML2D: Error reading %s\n", normalisation_filename);
		delete[] normalisation_filename;
		return Succeeded::no;
	      }

	  delete[] normalisation_filename;
	}
	// geo norm
      if (do_geo)
	{
	  {
	    char *normalisation_filename = new char[normalisation_filename_prefix.size() + 30];
	    sprintf(normalisation_filename, "%s_%s_%d_%d.out", 
		    normalisation_filename_prefix.c_str(), "geo", ax_pos_num, iter_num);
	    ifstream in(normalisation_filename);
	    in >> norm_geo_data;
	    if (!in)
	      {
		warning("BinNormalisationFromML2D: Error reading %s\n", normalisation_filename);
		delete[] normalisation_filename;
		return Succeeded::no;
	      }
	    delete[] normalisation_filename;
	  }
	}
	// block norm
      if (do_block)
	{
	  {
	    char *normalisation_filename = new char[normalisation_filename_prefix.size() + 30];
	    sprintf(normalisation_filename, "%s_%s_%d_%d.out", 
		    normalisation_filename_prefix.c_str(), "block", ax_pos_num, iter_num);
	    ifstream in(normalisation_filename);
	    in >> norm_block_data;
	    if (!in)
	      {
		warning("BinNormalisationFromML2D: Error reading %s\n", normalisation_filename);
		delete[] normalisation_filename;
		return Succeeded::no;
	      }
	    delete[] normalisation_filename;
	  }
	}
      {
	make_det_pair_data(det_pair_data, *proj_data_info_ptr, segment_num, ax_pos_num);
	det_pair_data.fill(1);
	if (do_eff)
	  apply_efficiencies(det_pair_data, efficiencies, true/*apply_or_undo*/);
	if (do_geo)
	  apply_geo_norm(det_pair_data, norm_geo_data, true/*apply_or_undo*/);
	if (do_block)
	  apply_block_norm(det_pair_data, norm_block_data, true/*apply_or_undo*/);
	set_det_pair_data(*norm_factors_ptr,
			  det_pair_data,
			  segment_num,
			  ax_pos_num);
      }
    }
 
 return Succeeded::yes;
}


void 
BinNormalisationFromML2D::apply(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams *= 
      norm_factors_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone());
  }

void 
BinNormalisationFromML2D::
undo(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams /= 
      norm_factors_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone());
  }

  
END_NAMESPACE_STIR

