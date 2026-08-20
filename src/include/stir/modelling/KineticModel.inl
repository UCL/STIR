/*
    Copyright (C) 2026, University Medical Center Groningen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

unsigned int
KineticModel::get_starting_frame() const
{
  return _starting_frame;
}

const TimeFrameDefinitions&
KineticModel::get_time_frame_definitions() const
{
  return _frame_defs;
}

int
KineticModel::get_frame_reference_time() const 
{
  return this->_frame_reference_time;
}

unsigned int
KineticModel::get_ending_frame() const
{
  return get_time_frame_definitions().get_num_frames();
}

const PlasmaData&
KineticModel::get_plasma_data() const
{
  return _plasma_frame_data; 
}

float 
KineticModel::get_calibration_factor() const 
{
  return _cal_factor; 
}

const shared_ptr<const ExamInfo>
KineticModel::get_exam_info_sptr() const
{
  return _exam_info_sptr;  
}

void
KineticModel::set_plasma_data(PlasmaData& arg)
{
  _already_setup = false; 
  _plasma_frame_data = arg; 
}

void
KineticModel::set_time_frame_definitions(TimeFrameDefinitions& arg)
{
  _already_setup = false; 
  _frame_defs = arg; 
}

void 
KineticModel::set_starting_frame(unsigned int arg)
{
  _already_setup = false;
  _starting_frame = arg;
}

void 
KineticModel::set_calibration_factor(float arg)
{
  _already_setup = false;
  _cal_factor = arg;
}

void 
KineticModel::set_exam_info(const shared_ptr<const ExamInfo>& exam_info_sptr)
{
  if (is_null_ptr(exam_info_sptr))
    error("KineticModel::set_exam_info: null exam info");
  _exam_info_sptr = exam_info_sptr; 
  set_radionuclide(exam_info_sptr->get_radionuclide()); 
}

void 
KineticModel::set_radionuclide(Radionuclide radionuclide_arg)
{
  _already_setup = false;
  this->_plasma_frame_data.set_isotope_halflife(radionuclide_arg.get_half_life()); 
}

void 
KineticModel::set_frame_reference_time(int arg)
{
  this->_frame_reference_time = arg; 
}

END_NAMESPACE_STIR