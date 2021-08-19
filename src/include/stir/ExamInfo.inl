/*
    Copyright (C) 2016, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief  This file declares the class stir::ExamInfo
  \author Kris Thielemans
  \author Nikos Efthimiou
*/

START_NAMESPACE_STIR

ExamInfo*
ExamInfo::clone() const
{
  return static_cast<ExamInfo*>(new ExamInfo(*this));
}

shared_ptr<ExamInfo>
ExamInfo::
create_shared_clone() const
{
  shared_ptr<ExamInfo> sptr(this->clone());
  return sptr;
}

void
ExamInfo::set_low_energy_thres(float new_val)
{
    low_energy_thres = new_val;
}

void
ExamInfo::set_high_energy_thres(float new_val)
{
    up_energy_thres = new_val;
}

void
ExamInfo::set_calibration_factor( const float cal_val)
{
    calibration_factor = cal_val;
}

void
ExamInfo::set_radionuclide(const std::string& name)
{
    radionuclide = name;
}

float
ExamInfo::get_low_energy_thres() const
{
    return low_energy_thres;
}

float
ExamInfo::get_high_energy_thres() const
{
    return up_energy_thres;
}

float
ExamInfo::get_calibration_factor() const
{
    return this->calibration_factor;
}

std::string
ExamInfo::get_radionuclide() const
{
    return radionuclide;
}

void
ExamInfo::set_energy_information_from(const ExamInfo& other)
{
  this->up_energy_thres = other.up_energy_thres;
  this->low_energy_thres = other.low_energy_thres;
}

END_NAMESPACE_STIR
