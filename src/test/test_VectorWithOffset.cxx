// $Id$
/*!

  \file
  \ingroup test

  \brief Test program for VectorWithOffset

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/VectorWithOffset.h"
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
#endif

START_NAMESPACE_STIR


/*!
  \brief Test class for VectorWithOffset
  \ingroup test

*/
class VectorWithOffsetTests : public RunTests
{
public:
  void run_tests();
};

void
VectorWithOffsetTests::run_tests()
{
  cerr << "Tests for VectorWithOffset\n"
       << "Everythings is fine if the program runs without any output." << endl;
  
  VectorWithOffset<int> v(-3, 40);

  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = 2*i;
  check_if_equal(v[4], 8, "test operator[]");

  int *ptr = v.get_data_ptr();
  ptr[4+3] = 5;
  v.release_data_ptr();
  check_if_equal(v[4], 5, "test get_data_tr/release_data_ptr");

  { 
    int value = -3;
    for (VectorWithOffset<int>::iterator iter = v.begin();
       iter != v.end();
       iter++, value++)
       *iter = value;
  }

  check_if_equal(v[4], 4, "test iterators");

  {
    int *p=find(v.begin(), v.end(), 6);
    check_if_equal(p - v.begin(), 9, "test iterators: find");
    check_if_equal(*p, 6, "test iterators: find");
  }

  sort(v.begin(), v.end(), greater<int>());
  check_if_equal(v[-3], 40, "test iterators: sort");
  check_if_equal(v[0], 37, "test iterators: sort");

#if 0
  // no reverse iterators yet...
  {
    VectorWithOffset<int>::reverse_iterator pr = v.rbegin();
    assert((*pr++) == 40);
    assert((*pr++) == 39);
  }

  sort(v.rbegin(), v.rend());
  check_if_equal(v[-3], 40, "test reverse iterators");
  check_if_equal(v[0], 37, "test reverse iterators");
#endif //0 (reverse iterators)

}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  VectorWithOffsetTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
