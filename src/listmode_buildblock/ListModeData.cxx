/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::ListModeData

  \author Daniel Deidda
  \author Kris Thielemans
*/
/*
    Copyright (C) 2003, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
    Copyright (C) 2019, National Physical Laboratory
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

#include "stir/listmode/ListModeData.h"
#include "stir/ExamInfo.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

ListModeData::
ListModeData()
{
}

ListModeData::
~ListModeData()
{}

const Scanner*
ListModeData::
get_scanner_ptr() const
{
    if(is_null_ptr(proj_data_info_sptr))
        error("ListModeData: ProjDataInfo has not been set.");
  return proj_data_info_sptr->get_scanner_ptr();
}

void
ListModeData::
set_proj_data_info_sptr(shared_ptr<const ProjDataInfo> new_proj_data_info_sptr)
{
    proj_data_info_sptr = new_proj_data_info_sptr;
}

shared_ptr<const ProjDataInfo>
ListModeData::get_proj_data_info_sptr() const
{
    if(is_null_ptr(proj_data_info_sptr))
        error("ListModeData: ProjDataInfo has not been set.");
    return proj_data_info_sptr;
}

#if 0
std::time_t
ListModeData::
get_scan_start_time_in_secs_since_1970() const
{
  const double time = this->exam_info_sptr->start_time_in_secs_since_1970;
  if (time<=0)
    return std::time_t(-1);
  else
    return time;
}
#endif

END_NAMESPACE_STIR
