//
//
/*
    Copyright (C) 1998- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018 University of Hull
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
  \ingroup listmode
  \brief Implementation of classes CListEventECAT966 and CListRecordECAT966 
  for listmode events for the ECAT 966 (aka Exact 3d).
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/

#include "stir/listmode/CListRecordECAT966.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

bool CListRecordECAT966::is_time() const
{ return this->time_data.is_time(); }

bool CListRecordECAT966::is_gating_input() const
{ return this->is_time(); }

bool CListRecordECAT966::is_event() const
{ return !this->is_time(); }

CListEventECAT966&  CListRecordECAT966::event()
  { return this->event_data; }

const CListEventECAT966&  CListRecordECAT966::event() const
  { return this->event_data; }

CListTimeECAT966&   CListRecordECAT966::time()
  { return this->time_data; }

const CListTimeECAT966&  CListRecordECAT966::time() const
  { return this->time_data; }

CListTimeECAT966&  CListRecordECAT966::gating_input()
  { return this->time_data; }

const CListTimeECAT966&  CListRecordECAT966::gating_input() const
{ return this->time_data; }


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
