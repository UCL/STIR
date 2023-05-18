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
#include "stir/Succeeded.h"
#include "stir/warning.h"
#include "stir/error.h"

START_NAMESPACE_STIR

void 
BinNormalisationWithCalibration::set_defaults()
{
  base_type::set_defaults();
  
  this->calibration_factor = 1;
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

Succeeded
BinNormalisationWithCalibration::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
                                        const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
{
  if (this->calibration_factor == 1.F)
    warning("BinNormalisationWithCalibration:: calibration factor not set. I will use 1, but your data will not be calibrated.");

  this->_calib_decay_branching_ratio = this->calibration_factor* this->get_branching_ratio(); //TODO: multiply by decay
  return base_type::set_up(exam_info_sptr, proj_data_info_sptr);
}


float 
BinNormalisationWithCalibration::
get_calib_decay_branching_ratio_factor(const Bin&) const
{
  if (!this->_already_set_up)
    error("BinNormalisationWithCalibration needs to be set-up first");
  return this->_calib_decay_branching_ratio;
}

float
BinNormalisationWithCalibration::
get_calibration_factor() const {
   return this->calibration_factor;
}

void
BinNormalisationWithCalibration::
set_calibration_factor(const float calib)
{
  this->_already_set_up = false;
  this->calibration_factor=calib;
}

float
BinNormalisationWithCalibration::
get_branching_ratio() const
{
  float branching_ratio = this->radionuclide.get_branching_ratio(false); // get value without check
  if (branching_ratio <=0)
    {
      warning("BinNormalisationWithCalibration: radionuclide branching_ratio not known. I will use 1.");
      return 1.F;
    }
   return branching_ratio;
}


void
BinNormalisationWithCalibration::
set_radionuclide(const Radionuclide &rnuclide){
    this->radionuclide=rnuclide;
}


END_NAMESPACE_STIR
