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
ExamInfo::set_low_energy_thres(float new_val,int en_window)
{
    low_energy_thres[en_window] = new_val;

}

void
ExamInfo::set_high_energy_thres(float new_val,int en_window)
{
    up_energy_thres[en_window] = new_val;

}

void
ExamInfo::set_low_energy_thres_vect(std::vector<float> new_val)
{
    low_energy_thres = new_val;
}

void
ExamInfo::set_high_energy_thres_vect(std::vector<float> new_val)
{
    up_energy_thres = new_val;
}

void
ExamInfo::set_num_energy_windows(int n_win)
{
    num_windows = n_win;

}


void
ExamInfo::set_energy_window_pair(std::vector<int> val)
{
   en_win_pair=val;
}


void
ExamInfo::set_calibration_factor( const float cal_val)
{
    calibration_factor = cal_val;
}

void
ExamInfo::set_radionuclide(const Radionuclide& arg)
{
    radionuclide = arg;
}

//Get the lower energy boundary for all the energy windows. en_window is set to 0 by default
//So that it will work also in the case of 1 energy window
float
ExamInfo::get_low_energy_thres(int en_window) const
{
    return low_energy_thres[en_window];

}

//Get the high energy boundary for all the energy windows. en_window is set to 0 by default
//So that it will work also in the case of 1 energy window

float
ExamInfo::get_high_energy_thres(int en_window) const
{

    return up_energy_thres[en_window];

}


//Get the number of energy windows
int
ExamInfo::get_num_energy_windows() const
{

    return num_windows;


}

std::pair<int,int>
ExamInfo::get_energy_window_pair() const
{

std::pair<int,int> pair;
pair.first = 0;
pair.second = 0;


pair.first=en_win_pair[0];
pair.second=en_win_pair[1];

    return pair;

}


float
ExamInfo::get_calibration_factor() const
{
    return this->calibration_factor;
}

Radionuclide
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
