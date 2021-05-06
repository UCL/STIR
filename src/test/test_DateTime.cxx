//
//
/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file 
  \ingroup test
  \ingroup date_time
  \brief A simple program to test the date-time conversions

  \author Kris Thielemans
*/
#include "stir/RunTests.h"
#include "stir/date_time_functions.h"

#include <iostream>


START_NAMESPACE_STIR

/*!
  \brief Class with tests for date time functions
  \ingroup test
  \ingroup date_time
*/
class DateTimeTest : public RunTests
{
  void check_round_trip(const double secs, const double tz_offset, const std::string& str);
public:
  void run_tests();
};


void
DateTimeTest::run_tests()
{
  // just do a consistency check first: mktime and local time should be "inverse" of eachother
  {
    time_t current_time = time(0);
    struct tm * local_time = localtime(&current_time);
    if (difftime(current_time, mktime(local_time)) != 0)
      error("CTIME internal error");
  }

  std::cerr << "Testing DICOM DateTime to epoch functionality\n";
  {
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("19700101000000.00+0000") -
                  0., "test 1 Jan 1970 is 0");
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("19710202000000+0000") -
                  (((((365 + 31 + 1)*24) + 0)*60. + 0)*60 + 0), "test 2 Feb 1971 0:0:");
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("19710202230001.80+0000") -
                  (((((365 + 31 + 1)*24) + 23)*60. + 0)*60 + 1.80), "test 2 Feb 1971 23:0:1.8");
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("19710202230301+0230") -
                  (((((365 + 31 + 1)*24) + 23 - 2.5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 +02:30");
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("19710202230301-0500") -
                  (((((365 + 31 + 1)*24) + 23 + 5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 -05:00");
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch(DICOM_date_time_to_DT("19710202", "230301", "-0500")) -
                  (((((365 + 31 + 1)*24) + 23 + 5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 -05:00 (split)");

    std::cerr << "\nThe next test should throw an error\n";
    try
      {
        DICOM_datetime_to_secs_since_Unix_epoch("19710202230301+020");
        check(false, "test ill-formed TZ");
      }
    catch (...)
      {
        std::cerr << "Test was ok\n";
      }

    // test difference, disabling warnings
    check_if_zero(DICOM_datetime_to_secs_since_Unix_epoch("20700104000000.4", true) -
                  DICOM_datetime_to_secs_since_Unix_epoch("20700101000000", true) -
                  (3*24*60.*60 + 0.4), "test difference without TZ");
  }

  std::cerr << "\nTesting Interfile DateTime to epoch functionality\n";
  {
    check_if_zero(Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1970:01:01", "00:00:00.00+0000")) -
                  0., "test 1 Jan 1970 is 0");
    check_if_zero(Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1971:02:02", "00:00:00+0000")) -
                  (((((365 + 31 + 1)*24) + 0)*60. + 0)*60 + 0), "test 2 Feb 1971 0:0:");
    check_if_zero(Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1971:02:02", "23:00:01.80+0000")) -
                  (((((365 + 31 + 1)*24) + 23)*60. + 0)*60 + 1.80), "test 2 Feb 1971 23:0:1.8");
    check_if_zero(Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1971:02:02", "23:03:01+0230")) -
                  (((((365 + 31 + 1)*24) + 23 - 2.5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 +02:30");
    check_if_zero(Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1971:02:02", "23:03:01-0500")) -
                  (((((365 + 31 + 1)*24) + 23 + 5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 -05:00");

    std::cerr << "\nThe next test should throw an error\n";
    try
      {
        Interfile_datetime_to_secs_since_Unix_epoch(DateTimeStrings("1971:02:2", "23:03:01"));
        check(false, "test ill-formed date");
      }
    catch (...)
      {
        std::cerr << "Test was ok\n";
      }

  }

  std::cerr << "\nTesting round-trip\n";
  {
    // a time in November
    double secs;

    secs = DICOM_datetime_to_secs_since_Unix_epoch("20201120223001.5+0000");
    check_round_trip(secs, 0., "round-trip 1 tz+0");
    check_round_trip(secs, 5.5*3600., "round-trip 1 tz+5.5");
    check_round_trip(secs, 12*3600., "round-trip 1 tz+12");
    check_round_trip(secs, -12*3600., "round-trip 1 tz-12");

    // a time in July (opposite DST situation)
    secs = DICOM_datetime_to_secs_since_Unix_epoch("20200720235901.5+0000");
    check_round_trip(secs, 0., "round-trip 2 tz+0");
    check_round_trip(secs, 5.5*3600., "round-trip 2 tz+5.5");
    check_round_trip(secs, 12*3600., "round-trip 2 tz+12");
    check_round_trip(secs, -12*3600., "round-trip 2 tz-12");
  }

  std::cerr << "\nCurrent timezone offset in hours (I cannot check this though):\n"
            << "  without DST: " << time_zone_offset_in_secs()/3600.
            << "  with DST: " << current_time_zone_and_DST_offset_in_secs()/3600.
            << "\n";

}

void
DateTimeTest::
check_round_trip(const double secs, const double tz_offset, const std::string& str)
{
  try
    {
      {
        const std::string time = secs_since_Unix_epoch_to_DICOM_datetime(secs, tz_offset);
        const double new_secs = DICOM_datetime_to_secs_since_Unix_epoch(time);
        check_if_zero(new_secs - secs, str + " : " +time);
      }
      {
        const DateTimeStrings dt = secs_since_Unix_epoch_to_Interfile_datetime(secs, tz_offset);
        const double new_secs = Interfile_datetime_to_secs_since_Unix_epoch(dt);
        check_if_zero(new_secs - secs, str + " : " + dt.date + ", " + dt.time);
      }
    }
  catch (...)
    {
      check(false, str);
    }
}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR



int main()
{
  DateTimeTest tests;
  tests.run_tests();
  return tests.main_return_value();
}
