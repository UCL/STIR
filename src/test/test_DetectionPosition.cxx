//
//

/*!
  \file 
 
  \brief A simple program to test the stir::DetectionPosition class

  \author Kris Thielemans

*/
/*
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/DetectionPosition.h"
#include "stir/RunTests.h"

using std::cerr;
using std::endl;

START_NAMESPACE_STIR

/*!
  \brief Class with tests for DetectionPosition.
*/
class DetectionPosition_Tests : public RunTests
{
public:
  void run_tests() override;
};


void
DetectionPosition_Tests::run_tests()
{
  cerr << "Testing DetectionPosition classes" << endl
       <<"  (There should be only informative messages here starting with 'Testing')" << endl;

  DetectionPosition<> pos012(0,1,2);
  DetectionPosition<> pos013(0,1,3);
  DetectionPosition<> pos023(0,2,3);
  DetectionPosition<> pos103(1,0,3);  

  check(pos012 != pos013, "012 != 013");
  check(pos012 == pos012, "012 == 012");
  check(pos012 < pos013, "012 < 013");
  check(pos012 < pos023, "012 < 023");
  check(pos012 < pos103, "012 < 103");
  check(pos103 > pos012, "103 > 012");
  check(pos012 <= pos013, "012 <= 013");
}

END_NAMESPACE_STIR



USING_NAMESPACE_STIR


int main()
{
  DetectionPosition_Tests tests;
  tests.run_tests();
  return tests.main_return_value();

}
