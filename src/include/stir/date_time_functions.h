/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file 
  \ingroup date_time
 
  \brief Functions for date-time conversions

  \author Kris Thielemans
*/

#ifndef __date_time_functions_H__
#define __date_time_functions_H__

#include "stir/common.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \brief returns the current time_zone in seconds (without DST)
  \ingroup date_time
  The result is independent of DST.
*/
int time_zone_offset_in_secs();

/*!
  \brief returns the current time_zone in seconds (taking DST into account)
  \ingroup date_time
  \see time_zone_offset_in_secs()
*/
int current_time_zone_and_DST_offset_in_secs();

/*!
  \brief concatenate date, time and optional time_zone info.
  \ingroup date_time

  Minimal checks on format are performed, calling \c error() if input is incorrect.
 */

std::string DICOM_date_time_to_DT(const std::string& date, const std::string& time, const std::string& TZ ="");

/*!
 \brief convert DICOM DT string to seconds since the Unix epoch (i.e. 1 Jan 1970 00:00:00 UTC)
 \ingroup date_time

  \arg str A DICOM date-time character string in the format:  YYYYMMDDHHMMSS.FFFFFF&ZZXX
  \arg silent if \c true, silence warnings about TZ info not being present
  Time_Zone info is given by "&ZZXX", with  & = "+" or "-", and ZZ = Hours and XX = Minutes of offset w.r.t. UTC.
  If no TZ is given, the local time_zone with DST is used. A warning() is then issued, unless \c silent=true.
*/
double DICOM_datetime_to_secs_since_Unix_epoch(const std::string& str, bool silent=false);

/*!
 \brief convert epoch to DICOM DT string in specified time zone (+3600 is CET)
 \ingroup date_time
*/
std::string
secs_since_Unix_epoch_to_DICOM_datetime(double secs,
                                        int time_zone_offset_in_secs = current_time_zone_and_DST_offset_in_secs());

/*! 
  \brief A simple structure to hold 2 strings (\c date and \c time)
 \ingroup date_time
*/
struct DateTimeStrings
{
  DateTimeStrings()
  {}
  DateTimeStrings(const std::string& date, const std::string& time)
  : date(date), time(time)
  {}

  std::string date, time;
};

//! Convert from DICOM DT to Interfile
/*! \ingroup date_time
 */
DateTimeStrings
DICOM_datetime_to_Interfile(const std::string& str);

//! Convert from Interfile to DICOM DT
/*! \ingroup date_time
 */
std::string
Interfile_datetime_to_DICOM(const DateTimeStrings&);

/*!
 \brief convert Interfile DateTime strings to seconds since the Unix epoch (i.e. 1 Jan 1970 00:00:00 UTC)
 \ingroup date_time
 Interfile uses a format year 1900:01:01, time 02:01:00.00.

 Uses DICOM conventions for specifying the time_zone, i.e. appending "&ZZXX", with
  & = "+" or "-", and ZZ = Hours and XX = Minutes of offset w.r.t. UTC.
*/
double Interfile_datetime_to_secs_since_Unix_epoch(const DateTimeStrings&, bool silent=false);

/*!
 \brief convert epoch to Interfile date-times string in specified time zone (+3600 is CET)
 \ingroup date_time
*/
DateTimeStrings
secs_since_Unix_epoch_to_Interfile_datetime(double secs,
                                            int time_zone_offset_in_secs = current_time_zone_and_DST_offset_in_secs());

END_NAMESPACE_STIR

#endif // __date_time_functions_H__
