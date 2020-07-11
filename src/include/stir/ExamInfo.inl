/*
    Copyright (C) 2016, University College London
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

END_NAMESPACE_STIR
