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
//#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
//#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
//#include "stir/RelatedViewgrams.h"
//#include "stir/Bin.h"
//#include "stir/ProjData.h"
//#include "stir/is_null_ptr.h"
//#include "stir/Succeeded.h"
//#include "stir/error.h"
//#include <boost/format.hpp>

START_NAMESPACE_STIR

void 
BinNormalisationWithCalibration::set_defaults()
{
  base_type::set_defaults();
  this->calibration_factor = 1;
 /* this->branching_ratio=1;*/
}

void 
BinNormalisationWithCalibration::
initialise_keymap()
{
  base_type::initialise_keymap();/*
  this->parser.add_key("calibration_factor", &this->calibration_factor);
  this->parser.add_key("branching_ratio", &this->branching_ratio);*/
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

void
BinNormalisationWithCalibration::
set_calibration_factor(const float calib){
    this->calibration_factor=calib;
}

void
BinNormalisationWithCalibration::
set_radionuclide(const std::string& rnuclide){
    this->radionuclide=rnuclide;
}
// TODO remove duplication between apply and undo by just having 1 functino that does the loops


END_NAMESPACE_STIR
