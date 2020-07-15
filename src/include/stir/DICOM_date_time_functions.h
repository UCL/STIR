//
//
/*
    Copyright (C) 2020, University College London
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
 
  \brief Functions for DICOM date-time conversions

  \author Kris Thielemans
*/

#ifndef __DICOM_data_time_functions_H__
#define __DICOM_data_time_functions_H__

#include "stir/common.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \brief concatenate date, time and optional timezone info.

  Minimal checks on format are performed, calling \c error() if input is incorrect.
 */

std::string DICOM_date_time_to_DT(const std::string& date, const std::string& time, const std::string& TZ ="");

/*!
 \brief convert DICOM DT string to seconds since epoch (i.e. 1 Jan 1970 00:00:00 UTC)

  \arg str    A DICOM date-time character string in the format:  YYYYMMDDHHMMSS.FFFFFF&ZZXX

  TimeZone info is given by "&ZZXX", with  & = "+" or "-", and ZZ = Hours and XX = Minutes of offset w.r.t. UTC.
  If no TZ is given, the local timezone with DST is used. A warning() is then issued, unless \c silent=true.
*/
double DICOM_datetime_to_secs_since_epoch(const std::string& str, bool silent=false);

END_NAMESPACE_STIR

#endif // __DICOM_data_time_functions_H__
