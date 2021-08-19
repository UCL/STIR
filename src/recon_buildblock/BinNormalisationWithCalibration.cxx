//
//
/*
    Copyright (C) 2020, National Physical Laboratory
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
  base_type::initialise_keymap();/*
  this->parser.add_key("calibration_factor", &this->calibration_factor);
  this->parser.add_key("branching_ratio", &this->branching_ratio);*/
}

bool 
BinNormalisationWithCalibration::
post_processing()
{
  return base_type::post_processing();
}


BinNormalisationWithCalibration::
BinNormalisationWithCalibration()
{
  set_defaults();
}

float 
BinNormalisationWithCalibration::
get_calib_decay_branching_ratio_factor(const Bin&) const{
    return this->calibration_factor* this->branching_ratio; //TODO: multiply by branching factor and decay
}

float
BinNormalisationWithCalibration::
get_calibration_factor() const {
   return this->calibration_factor;
}

void
BinNormalisationWithCalibration::
set_calibration_factor(const float calib){
    this->calibration_factor=calib;
}

float
BinNormalisationWithCalibration::
get_branching_ratio() const {
   return this->branching_ratio;
}

void
BinNormalisationWithCalibration::
set_branching_ratio(const float br){
    this->branching_ratio=br;
}

void
BinNormalisationWithCalibration::
set_radionuclide(const std::string& rnuclide){
    this->radionuclide=rnuclide;
}


END_NAMESPACE_STIR
