//
// $Id$: $Date$
//

/*!
  \file 
 
  \brief A simple programme to test the IndexRange class

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

#include "IndexRange.h"
#include "IndexRange2D.h"
#include "IndexRange3D.h"
#include "Coordinate3D.h"
#include "RunTests.h"

START_NAMESPACE_TOMO

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

}

END_NAMESPACE_TOMO



USING_NAMESPACE_TOMO


int main()
{
  IndexRange_Tests tests;
  tests.run_tests();
  return tests.main_return_value();

}
