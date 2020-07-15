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
  \ingroup test
 
  \brief A simple program to test the DICOM date-time conversions

  \author Kris Thielemans
*/
#include "stir/RunTests.h"
#include "stir/DICOM_date_time_functions.h"

#include <iostream>


START_NAMESPACE_STIR

/*!
  \brief Class with tests for DICOM date time functions
  \ingroup test
*/
class DICOMDateTimeTest : public RunTests
{
public:
  void run_tests();
};


void
DICOMDateTimeTest::run_tests()
{
  std::cerr << "Testing DICOM DateTime functionality\n";

  check_if_zero(DICOM_datetime_to_secs_since_epoch("19700101000000.00+0000") - 
                0., "test 1 Jan 1970 is 0");
  check_if_zero(DICOM_datetime_to_secs_since_epoch("19710202000000+0000") -
                (((((365 + 31 + 1)*24) + 0)*60. + 0)*60 + 0), "test 2 Feb 1971 0:0:");
  check_if_zero(DICOM_datetime_to_secs_since_epoch("19710202230001.80+0000") -
                (((((365 + 31 + 1)*24) + 23)*60. + 0)*60 + 1.80), "test 2 Feb 1971 23:0:1.8");
  check_if_zero(DICOM_datetime_to_secs_since_epoch("19710202230301+0230") -
                (((((365 + 31 + 1)*24) + 23 - 2.5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 +02:30");
  check_if_zero(DICOM_datetime_to_secs_since_epoch("19710202230301-0500") -
                (((((365 + 31 + 1)*24) + 23 + 5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 -05:00");
  check_if_zero(DICOM_datetime_to_secs_since_epoch(DICOM_date_time_to_DT("19710202", "230301", "-0500")) -
                (((((365 + 31 + 1)*24) + 23 + 5)*60. + 3)*60 + 1), "test 2 Feb 1971 23:03:01 -05:00 (split)");

  std::cerr << "\nThe next test should throw an error\n";
  try
    {
      DICOM_datetime_to_secs_since_epoch("19710202230301+020");
      check(false, "test ill-formed TZ");
    }
  catch (...)
    {
      // ok
    }
  
  // test difference, disabling warnings
  check_if_zero(DICOM_datetime_to_secs_since_epoch("20700104000000.4", true) -
                DICOM_datetime_to_secs_since_epoch("20700101000000", true) - 
                (3*24*60.*60 + 0.4), "test difference without TZ");

}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR



int main()
{
  DICOMDateTimeTest tests;
  tests.run_tests();
  return tests.main_return_value();
}
