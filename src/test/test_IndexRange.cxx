//
//

/*!
  \file 
 
  \brief A simple program to test the stir::IndexRange class

  \author Kris Thielemans
  \author PARAPET project



*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/Coordinate3D.h"
#include "stir/RunTests.h"

START_NAMESPACE_STIR

/*!
  \brief Class with tests for IndexRange, IndexRange3D.
*/
class IndexRange_Tests : public RunTests
{
public:
  void run_tests();
};


void
IndexRange_Tests::run_tests()
{
  cerr << "Testing IndexRange classes" << endl
       <<"  (There should be only informative messages here starting with 'Testing')" << endl;

  // make an irregular range
  {
    IndexRange<1> range1(1,3);
    IndexRange<1> range2(2,4);
    VectorWithOffset< IndexRange<1> > range2d(3,4);
    range2d[3]=range1;
    range2d[4]=range2;
    IndexRange<2> idx_range2d = range2d;
    
    check(idx_range2d[3].get_min_index() == range1.get_min_index(), 
      "testing constructor from base_type");
    check(idx_range2d[3].get_max_index() == range1.get_max_index(), 
      "testing constructor from base_type");
    check(idx_range2d.is_regular()==false, 
      "testing is_regular on irregular range");
  }
  // make a regular range
  {
    IndexRange<1> range1(1,3);
    VectorWithOffset< IndexRange<1> > range2d(3,4);
    range2d[3]=range1;
    range2d[4]=range1;
    IndexRange<2> idx_range2d = range2d;
    
    check(idx_range2d[3].get_min_index() == range1.get_min_index(), 
      "testing constructor from base_type");
    check(idx_range2d[3].get_max_index() == range1.get_max_index(), 
      "testing constructor from base_type");
    check(idx_range2d.is_regular()==true, 
      "testing is_regular on irregular range");

    IndexRange2D another_idx_range2d(3,4, 1,3);
    check(another_idx_range2d == idx_range2d, "test IndexRange2D");
  }

  {
    Coordinate3D<int> low(1,2,3);
    Coordinate3D<int> high(3,4,5);
    IndexRange<3> idx_range3d(low, high);
    
    check(idx_range3d[3].get_max_index() == 4, 
      "testing constructor from 2 Coordinate objects");
    check(idx_range3d.is_regular()==true, 
      "testing is_regular on regular range");
    Coordinate3D<int> low_test, high_test;
    if (idx_range3d.get_regular_range(low_test, high_test))
    {
      check_if_equal(low, low_test, "testing is_regular on regular range: lower indices");
      check_if_equal(high, high_test, "testing is_regular on regular range: higher indices");
    }

    IndexRange3D another_idx_range3d(low[1],high[1], low[2], high[2], low[3], high[3]);
    check(another_idx_range3d == idx_range3d, "test IndexRange3D");
  }
  {
    const Coordinate3D<int> sizes(3,4,5);
    const IndexRange<3> idx_range3d(sizes);
    
    check(idx_range3d.get_max_index() == 2, 
      "testing constructor from 1 Coordinate object");
    check(idx_range3d[0].get_max_index() == 3, 
      "testing constructor from 1 Coordinate object");
    check(idx_range3d[0][0].get_max_index() == 4, 
      "testing constructor from 1 Coordinate object");
    check(idx_range3d.is_regular()==true, 
      "testing is_regular on regular range");
    Coordinate3D<int> low_test, high_test;
    if (idx_range3d.get_regular_range(low_test, high_test))
    {
      check_if_equal(Coordinate3D<int>(0,0,0), low_test, "testing is_regular on regular range: lower indices");
      check_if_equal(sizes-1, high_test, "testing is_regular on regular range: higher indices");
    }

    const IndexRange3D another_idx_range3d(sizes[1], sizes[2], sizes[3]);
    check(another_idx_range3d == idx_range3d, "test IndexRange3D");
  }

}

END_NAMESPACE_STIR



USING_NAMESPACE_STIR


int main()
{
  IndexRange_Tests tests;
  tests.run_tests();
  return tests.main_return_value();

}
