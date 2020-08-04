/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-01 - 2013, Kris Thielemans
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

  \brief Test program for stir::NumericVectorWithOffset

  \author Kris Thielemans
  \author PARAPET project


*/
#ifndef NDEBUG
// set to high level of debugging
#ifdef _DEBUG
#undef _DEBUG
#endif
#define _DEBUG 2
#endif

#include "stir/NumericVectorWithOffset.h"
#include "stir/RunTests.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::sort;
using std::find;
using std::greater;
using std::size_t;
#endif

START_NAMESPACE_STIR


/*!
  \brief Test class for NumericVectorWithOffset
  \ingroup test

*/
class NumericVectorWithOffsetTests : public RunTests
{
public:
  void run_tests();
};

void
NumericVectorWithOffsetTests::run_tests()
{
  cerr << "Tests for NumericVectorWithOffsetTests\n"
       << "Everythings is fine if the program runs without any output." << endl;
  
  NumericVectorWithOffset<float,float> test(-3, 40);
  test.fill(1.);
  /**********************************************************************/
  // tests on xapyb
  /**********************************************************************/
  {
    NumericVectorWithOffset<float,float> tmp(test);
    NumericVectorWithOffset<float,float> tmp2(test+2);
    tmp.axpby(2.F, test, 3.3F, tmp2);
    NumericVectorWithOffset<float,float> by_hand = test*2.F + (test+2)*3.3F;
    check_if_equal(tmp, by_hand, "test axpby");    
  }
  {
    NumericVectorWithOffset<float,float> tmp(test);
    NumericVectorWithOffset<float,float> tmp2(test+2);
    tmp.xapyb(test, 2.F, tmp2, 3.3F);
    NumericVectorWithOffset<float,float> by_hand = test*2.F + (test+2)*3.3F;
    check_if_equal(tmp, by_hand, "test xapyb");    
  }
  {
    NumericVectorWithOffset<float,float> tmp(test);
    NumericVectorWithOffset<float,float> tmp2(test+2);
    NumericVectorWithOffset<float,float> tmpa(test+4);
    NumericVectorWithOffset<float,float> tmpb(test+6);

    tmp.xapyb(test, tmpa, tmp2, tmpb);
    NumericVectorWithOffset<float,float> by_hand = test*(test+4) + (test+2)*(test+6);
    check_if_equal(tmp, by_hand, "test xapyb where a,b are vectors");
  }  
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  NumericVectorWithOffsetTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
