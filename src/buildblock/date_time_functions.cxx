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
#include "stir/date_time_functions.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/round.h"
#include "boost/lexical_cast.hpp"
#include "stir/format.h"
#include <string>

START_NAMESPACE_STIR

int
time_zone_offset_in_secs()
{
  static bool first_run = true;
  static int tz_offset;

  if (first_run)
    {
      first_run = false;

      time_t current_time = time(0);
      // struct tm * local_time = localtime(&current_time);
      // std::cerr << "Local: " << local_time->tm_hour << ',' << local_time->tm_isdst;
      struct tm* gmt = gmtime(&current_time);
      // std::cerr << ", GMT: " << gmt->tm_hour << ',' << gmt->tm_isdst << "\n";
      time_t gm_time = mktime(gmt);
      // std::cerr << " diff Local-GMT: " << difftime(current_time, gm_time)/3600. << "\n";
      tz_offset = round(difftime(current_time, gm_time));
    }
  return tz_offset;
}

int
current_time_zone_and_DST_offset_in_secs()
{
  time_t current_time = time(0);
  struct tm* local_time = localtime(&current_time);
  const int isdst = local_time->tm_isdst;
  return time_zone_offset_in_secs() + isdst * 3600;
}

/* internal function to find the time_t for the Unix epoch
  This is usually zero, but it is not guaranteed.

  KT tested this on Ubuntu 18.04 on multiple time_zones, and it was 0 except
  in the GB/Portugal time_zone. This is of course weird.
  However, as long as we handle this internally consistently, it shouldn't matter.
*/
static time_t
unix_epoch_time_t()
{
  static bool first_run = true;
  static time_t epoch_offset;

  if (first_run)
    {
      first_run = false;
      struct tm time_info_start; // 1 JAN 1970 00:00 in local time (without DST)
      time_info_start.tm_year = 1970 - 1900;
      time_info_start.tm_mon = 0;
      time_info_start.tm_mday = 1;
      time_info_start.tm_sec = 0;
      time_info_start.tm_min = 0;
      time_info_start.tm_hour = 0;
      time_info_start.tm_isdst = 0;
      time_t loc_time_start = mktime(&time_info_start);
      epoch_offset = loc_time_start + time_zone_offset_in_secs();
      if (epoch_offset != 0)
        info(format("Using Unix epoch (1Jan1970) offset of {}", epoch_offset), 3);
    }

  return epoch_offset;
}

std::string
DICOM_date_time_to_DT(const std::string& date_org, const std::string& time_org, const std::string& TZ_org)
{
  // get rid of white spaces, just in case
  const std::string date = standardise_interfile_keyword(date_org);
  const std::string time = standardise_interfile_keyword(time_org);
  const std::string TZ = standardise_interfile_keyword(TZ_org);
  if ((date.size() != 8) || (time.size() < 6 || (time.size() > 6 && time[6] != '.')) || (!TZ.empty() && TZ.size() != 5))
    error(format("DICOM_date_time_to_DT: ill-formed input: date={}, time={}, TZ info={}", date, time, TZ));
  return date + time + TZ;
}

static double
parse_DICOM_TZ(const std::string& tz, const bool silent)
{
  double tz_offset;
  if (tz.empty())
    {
      {
        tz_offset = time_zone_offset_in_secs();
        if (!silent)
          warning(format("No Time_Zone info in DICOM DT. Using local time-zone without DST ({:+.0f} secs)", tz_offset));
      }
    }
  else
    {
      if (tz.size() != 5)
        error("Time_Zone info '" + tz + "' does not fit DICOM standard");
      else
        {
          tz_offset = (boost::lexical_cast<double>(tz.substr(0, 3)) * 60 + boost::lexical_cast<double>(tz.substr(3))) * 60;
          // info(format("Found time zone difference in DICOM DT '{}' of {} secs", str, tz_offset), 2);
        }
    }
  return tz_offset;
}

static void
parse_DICOM_fraction_and_TZ(std::string& fraction, std::string& tz_string, const std::string& str)
{
  const std::string rest = str.substr(14);

  std::size_t tz_pos = rest.find('+');
  if (tz_pos == std::string::npos)
    tz_pos = rest.find('-');
  if (tz_pos != std::string::npos)
    tz_string = rest.substr(tz_pos);
  if (tz_pos != std::string::npos)
    fraction = rest.substr(0, tz_pos);
  else
    fraction = rest;
}

double
DICOM_datetime_to_secs_since_Unix_epoch(const std::string& str_org, bool silent)
{
  // get rid of white spaces, just in case
  const std::string str = standardise_interfile_keyword(str_org);

  if (str.size() < 14)
    error("DICOM DT '" + str + "' is ill-formed");

  struct tm time_info;
  time_info.tm_year = boost::lexical_cast<int>(str.substr(0, 4)) - 1900;
  time_info.tm_mon = boost::lexical_cast<int>(str.substr(4, 2)) - 1;
  time_info.tm_mday = boost::lexical_cast<int>(str.substr(6, 2));
  time_info.tm_hour = boost::lexical_cast<int>(str.substr(8, 2));
  ;
  time_info.tm_min = boost::lexical_cast<int>(str.substr(10, 2));
  ;
  time_info.tm_sec = boost::lexical_cast<int>(str.substr(12, 2));
  time_info.tm_isdst = 0; // no DST
  // find the time as if the above is specified in the local time_zone
  double time_diff = difftime(mktime(&time_info), unix_epoch_time_t());

  /* add fraction and time_zone */
  {
    std::string tz_string;
    std::string fraction;
    parse_DICOM_fraction_and_TZ(fraction, tz_string, str);

    const double tz_offset = parse_DICOM_TZ(tz_string, silent);
    // now go to time in UTC
    time_diff -= tz_offset - time_zone_offset_in_secs();

    // handle fraction of seconds
    if (fraction.size() > 0)
      {
        if (fraction[0] != '.')
          error("DICOM DT '" + str + "' is ill-formed for the fractional seconds");
        try
          {
            const double frac_secs = boost::lexical_cast<double>(fraction);
            time_diff += frac_secs;
          }
        catch (...)
          {
            error("DICOM DT '" + str + "' is ill-formed for the fractional seconds");
          }
      }
  }
  info(format("DICOM DT '{}' = {:.2f}s since unix epoch (1970)", str, time_diff), 3);
  return time_diff;
}

std::string
secs_since_Unix_epoch_to_DICOM_datetime(double secs, int time_zone_offset_in_secs)
{
  const int tz_in_mins = time_zone_offset_in_secs / 60;
  // check it's in minutes (as expected, but also imposed by DICOM)
  {
    if (round(tz_in_mins * 60 - secs) > 1)
      error(format("secs_since_Unix_epoch_to_DICOM_datetime: can only handle time_zone offsets that are a multiple of 60, "
                   "argument was {}",
                   time_zone_offset_in_secs));
  }

  time_t time = round(floor(secs) + unix_epoch_time_t() + time_zone_offset_in_secs);
  struct tm* time_info = gmtime(&time);
  return (format("{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}.{:02d}{:+03d}{:02d}",
                 (time_info->tm_year + 1900),
                 (time_info->tm_mon + 1),
                 time_info->tm_mday,
                 time_info->tm_hour,
                 time_info->tm_min,
                 time_info->tm_sec,
                 round((secs - floor(secs)) * 100),
                 (tz_in_mins / 60),
                 (tz_in_mins % 60)));
}

DateTimeStrings
DICOM_datetime_to_Interfile(const std::string& str)
{
  // just do a conversion to check on format (will throw if there's an error)
  DICOM_datetime_to_secs_since_Unix_epoch(str);
  DateTimeStrings dt;
  dt.date = str.substr(0, 4) + ':' + str.substr(4, 2) + ':' + str.substr(6, 2);
  dt.time = str.substr(8, 2) + ':' + str.substr(10, 2) + ':' + str.substr(12);
  return dt;
}

std::string
Interfile_datetime_to_DICOM(const DateTimeStrings& dt)
{
  // get rid of white spaces, just in case
  const std::string date = standardise_interfile_keyword(dt.date);
  const std::string time = standardise_interfile_keyword(dt.time);

  if ((date.size() != 10) || (date[4] != ':') || (date[7] != ':'))
    error("Interfile_datetime_to_DICOM: ill-formed date: " + date);
  if ((time.size() < 8) || (time[2] != ':') || (time[5] != ':'))
    error("Interfile_datetime_to_DICOM: ill-formed time: " + time);

  return date.substr(0, 4) + date.substr(5, 2) + date.substr(8, 2) + time.substr(0, 2) + time.substr(3, 2) + time.substr(6);
}

double
Interfile_datetime_to_secs_since_Unix_epoch(const DateTimeStrings& intf, bool silent)
{
  return DICOM_datetime_to_secs_since_Unix_epoch(Interfile_datetime_to_DICOM(intf), silent);
}

DateTimeStrings
secs_since_Unix_epoch_to_Interfile_datetime(double secs, int time_zone_offset_in_secs)
{
  return DICOM_datetime_to_Interfile(secs_since_Unix_epoch_to_DICOM_datetime(secs, time_zone_offset_in_secs));
}

END_NAMESPACE_STIR
