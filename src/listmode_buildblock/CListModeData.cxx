/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::CListModeData 
    
  \author Kris Thielemans
*/
/*
    Copyright (C) 2003, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
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

#include "stir/listmode/CListModeData.h"
#include "stir/ExamInfo.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

CListModeData::
CListModeData()
{
}

CListModeData::
~CListModeData()
{}

const ExamInfo*
CListModeData::get_exam_info_ptr() const
{
  assert(!is_null_ptr(exam_info_sptr));
  return exam_info_sptr.get();
}

shared_ptr<ExamInfo>
CListModeData::get_exam_info_sptr() const
{
  return exam_info_sptr;
}

const Scanner* 
CListModeData::
get_scanner_ptr() const
{
  assert(!is_null_ptr(exam_info_sptr));
  return this->scanner_sptr.get();
}

#if 0
std::time_t 
CListModeData::
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
