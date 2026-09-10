//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling
  \brief Implementations of inline functions of class stir::KineticModel

  \author Charalampos Tsoumpas

  This is the most basic class for including kinetic models.

*/

#include "stir/modelling/KineticModel.h"

START_NAMESPACE_STIR

const char* const KineticModel::registered_name = "Kinetic Model Type";

void
KineticModel::initialise_keymap()
{
  this->parser.add_key("frame reference time", &this->_frame_reference_time);
  this->parser.add_key("Blood Data Filename", &this->_blood_data_filename);
  this->parser.add_key("Calibration Factor", &this->_cal_factor);
  this->parser.add_key("Starting Frame", &this->_starting_frame);
  this->parser.add_key("Time Shift", &this->_time_shift);
  this->parser.add_key("In total counts", &this->_in_total_cnt);
  this->parser.add_key("In correct scale", &this->_in_correct_scale);
  this->parser.add_key("Time Frame Definition Filename", &this->_time_frame_definition_filename);
}

void
KineticModel::set_defaults()
{
  _blood_data_filename = "";
  _cal_factor = 1.F;
  _starting_frame = 0;
  _time_shift = 0.;
  _in_correct_scale = false;
  _in_total_cnt = false;
  _plasma_in_total_cnt = false;
  _already_setup = false;
  _matrix_is_stored = false;
  _frame_reference_time = 0;
}

Succeeded
KineticModel::set_up()
{
  _already_setup = true;
  return Succeeded::yes;
}

bool
KineticModel::post_processing()
{
  // read time frame def
  if (this->_time_frame_definition_filename.size() != 0)
    _frame_defs = TimeFrameDefinitions(this->_time_frame_definition_filename);
  else
    {
      error("No Time Frames Definitions available!!!");
      return true;
    }

  // Reading the input function
  if (this->_blood_data_filename == "0")
    {
      warning("You need to specify a file for the input function.");
      return true;
    }
  else
    {
      this->_if_cardiac = false;
    }

  return false;
}

void
KineticModel::create_model_matrix()
{
  if (!_already_setup)
    error("You need to run set_up first!");

  if (this->_plasma_frame_data.get_is_decay_corrected())
    warning("Uncorrected previous decay correction, while putting the plasma_data into the model_matrix.");
  else
    error("plasma_data have not been corrected during the process, which will create wrong results!!!");
}

END_NAMESPACE_STIR
