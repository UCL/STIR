//
//
/*
    Copyright (C) 2020, National Physical Laboratory
    Copyright (C) 2020 University College London
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
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationWithCalibration

  \author Daniel Deidda
  \author Kris Thielemans
*/


#include "stir/recon_buildblock/BinNormalisationWithCalibration.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Bin.h"
#include "stir/ProjData.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

void 
BinNormalisationWithCalibration::set_defaults()
{
  base_type::set_defaults();
  this->calibration_factor = 1;
  this->branching_ratio=1;
}

void 
BinNormalisationWithCalibration::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_key("calibration_factor", &this->calibration_factor);
  this->parser.add_key("branching_ratio", &this->branching_ratio);
}

bool 
BinNormalisationWithCalibration::
post_processing()
{
  return base_type::post_processing();;
}


BinNormalisationWithCalibration::
BinNormalisationWithCalibration()
{
  set_defaults();
}

// TODO remove duplication between apply and undo by just having 1 functino that does the loops

void 
BinNormalisationWithCalibration::apply(RelatedViewgrams<float>& viewgrams,
			const double start_time, const double end_time) const 
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num()){
//          if (use_decay_correction_factors()){
//              normalisation=
//              normalisation/decay_correction_factor(half_life, rel_time);
//          }
        (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
          std::max(1.E-20F, get_bin_efficiency(bin, start_time, end_time)*this->calibration_factor/this->branching_ratio);}
  }
}         

void 
BinNormalisationWithCalibration::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  this->check(*viewgrams.get_proj_data_info_sptr());
//    double  start_time=
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= 
	   this->get_bin_efficiency(bin,start_time, end_time)*this->calibration_factor/this->branching_ratio;
  }

}
END_NAMESPACE_STIR

