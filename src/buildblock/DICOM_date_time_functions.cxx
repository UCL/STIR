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
#include "stir/DICOM_date_time_functions.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "boost/lexical_cast.hpp"
#include "boost/format.hpp"
#include <string>

START_NAMESPACE_STIR

std::string DICOM_date_time_to_DT(const std::string& date_org, const std::string& time_org, const std::string& TZ_org)
{
  // get rid of white spaces, just in case
  const std::string date = standardise_interfile_keyword(date_org);
  const std::string time = standardise_interfile_keyword(time_org);
  const std::string TZ = standardise_interfile_keyword(TZ_org);
  if ((date.size()!=8) || (time.size()<6 || (time.size()>6 && time[6]!='.'))
      || (!TZ.empty() && TZ.size()!=5))
    error(boost::format("DICOM_date_time_to_DT: ill-formed input: date=%s, time=%s, TZ info=%s") % date % time % TZ);
  return date+time+TZ;
}

double DICOM_datetime_to_secs_since_epoch(const std::string& str_org, bool silent)
{
  // get rid of white spaces, just in case
  const std::string str = standardise_interfile_keyword(str_org);

  if (str.size()<14)
    error("DICOM DT '" + str + "' is ill-formed");

  try
    {
      struct tm time_info;
      time_info.tm_year = boost::lexical_cast<int>(str.substr(0,4)) - 1900;
      time_info.tm_mon = boost::lexical_cast<int>(str.substr(4,2)) - 1;
      time_info.tm_mday = boost::lexical_cast<int>(str.substr(6,2));
      time_info.tm_hour = boost::lexical_cast<int>(str.substr(8,2));;
      time_info.tm_min = boost::lexical_cast<int>(str.substr(10,2));;
      time_info.tm_sec = boost::lexical_cast<int>(str.substr(12,2));
      time_info.tm_isdst = 0; // no DST
      struct tm time_info_start; // 1 JAN 1970 00:00
      time_info_start.tm_year = 1970 - 1900;
      time_info_start.tm_mon = 0;
      time_info_start.tm_mday = 1;
      time_info_start.tm_sec = 0;
      time_info_start.tm_min = 0;
      time_info_start.tm_hour = 0;
      time_info_start.tm_isdst = 0;

      const time_t loc_time_start = mktime(&time_info_start);
      // time difference from 1 Jan 1970 in local timezone
      double time_diff = difftime(mktime(&time_info), loc_time_start);

      // convert to time diff in UTC

      /* add fraction and timezone */
      {
        std::string rest = str.substr(14);
        // std::cerr << "Remaining: " <<rest << '\n';
        // find TZ string
        std::size_t tz_pos = rest.find('+');
        if (tz_pos == std::string::npos)
          tz_pos = rest.find('-');
        if (tz_pos == std::string::npos)
          {
            {
              struct tm * gtm = gmtime(&loc_time_start);
              time_t gm_time = mktime(gtm);
              double timezone_diff = difftime(loc_time_start, gm_time);
              time_diff -= timezone_diff;
              if (!silent)
                warning(boost::format("No TimeZone info in DICOM DT %s. Using local time-zone without DST (%+.0f secs)") % str % timezone_diff);
            }
          }
        else
          {
            if (rest.size() != tz_pos + 5)
              error("TimeZone info '" + rest.substr(tz_pos) + "' does not fit DICOM standard");
            else
              {
                const double tz_offset =
                  (boost::lexical_cast<double>(rest.substr(tz_pos,3))*60 +
                   boost::lexical_cast<double>(rest.substr(tz_pos+3)))*60;
                time_diff -= tz_offset;
                //info(boost::format("Found time zone difference in DICOM DT '%s' of %g secs")
                //     % str % tz_offset, 2);
              }
          }
        if (rest.size()>0 && rest[0] == '.')
          {
            const double frac_secs = boost::lexical_cast<double>(rest.substr(0, tz_pos));
            time_diff += frac_secs;
          }
      }
      info(boost::format("DICOM DT '%s' = %.2f")% str % time_diff, 2);
      return time_diff;
    }
  catch (...)
    {
      error("DICOM DT '" + str + "' is ill-formed");
      // never get here but avoid compiler warning
      return -1.;
    }
}


END_NAMESPACE_STIR
