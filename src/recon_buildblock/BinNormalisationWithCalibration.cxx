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
    return this->calibration_factor; //TODO: multiply by branching factor and decay
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

void
BinNormalisationWithCalibration::
set_radionuclide(const std::string& rnuclide){
    this->radionuclide=rnuclide;
}


END_NAMESPACE_STIR
