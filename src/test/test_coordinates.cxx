//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup test
 
  \brief A simple programme to test the Coordinate classes

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

#include "CartesianCoordinate3D.h"
#include "RunTests.h"
#include <iostream>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

/*!
  \brief Class with tests for BasicCoordinate, Coordinate3D et al.
  \ingroup test
*/
class coordinateTests : public RunTests
{
public:
  void run_tests();
};


void
coordinateTests::run_tests()
{
  cerr << "Testing Coordinate classes" << endl
       <<"  (There should be only informative messages here starting with 'Testing')" << endl;

  {
    cerr << "Testing BasicCoordinate<3,float>" << endl;

    BasicCoordinate<3, float> a;
    a[1]=1;a[2]=2;a[3]=3;
    BasicCoordinate<3, float> copy_of_a;
    copy_of_a[1]=1;copy_of_a[2]=2;copy_of_a[3]=3;
    BasicCoordinate<3, float> b;
    b[1]=-1;b[2]=-3;b[3]=5;
    BasicCoordinate<3, float> a_plus_b;
    a_plus_b[1]=0;a_plus_b[2]=-1;a_plus_b[3]=8;

    
    check(a[3]==3, "testing operator[]");
    check_if_equal(inner_product(a,b), 8., "testing inner_product");
    check_if_equal(norm(a), 3.74166, "testing norm");

    a += b;
    check_if_zero(a- a_plus_b, "testing operator+=(BasicCoordinate)");
    a -= b;
    check_if_equal(a, copy_of_a, "testing operator-=(BasicCoordinate)");
    
    {
      BasicCoordinate<3, float> b1 = b;
      check_if_zero(norm(b1-b), "testing copy constructor, and operator-");
      
      b1 = a;
      check_if_zero(norm(a-b1), "testing assignment");
    }

    a *= 4;
    check(a[1] == copy_of_a[1]*4, "testing operator*=(float)");
    check_if_equal(norm(a),  norm(copy_of_a)*4, "testing operator*=(float)");
    a /= 4;
    check_if_zero(norm(a-copy_of_a), "testing operator/=(float)");    

    {
      BasicCoordinate<3, float> a1;
      a1 = b;
      a1 *= 3;
      a1 += a;
      a1 -= 4;
      BasicCoordinate<3, float> a2 = (b*3+a)-4;
      check_if_zero(norm(a1-a2), "testing various numerical operators");    
    }

    // basic iterator tests
    { 
#ifndef TOMO_NO_NAMESPACES
      float *p=std::find(b.begin(), b.end(), -3);
#else
      float *p=find(b.begin(), b.end(), -3);
#endif
      check_if_equal(p - b.begin(), 1, "iterator test");
      BasicCoordinate<3, float> b_sorted;
      b_sorted[1]=-3;b_sorted[2]=-1;b_sorted[3]=5;
#ifndef TOMO_NO_NAMESPACES
      std::sort(b.begin(), b.end()); 
#else
      sort(b.begin(), b.end()); 
#endif
      check_if_zero(norm(b-b_sorted), "testing iterators via STL sort");
    }
    
    // join
#if !defined( __GNUC__) || !(__GNUC__ == 2 && __GNUC_MINOR__ < 9)
    {
      BasicCoordinate<3, float> a;
      a[1]=1;a[2]=2;a[3]=3;
      BasicCoordinate<4, float> a4 = join(0.F, a);
      check_if_equal(a4[1], 0., "testing join of float with BasicCoordinate");
      check_if_equal(a4[2], 1., "testing join of float with BasicCoordinate");
      check_if_equal(a4[3], 2., "testing join of float with BasicCoordinate");
      check_if_equal(a4[4], 3., "testing join of float with BasicCoordinate");
    }
#endif
  }

  // essentially the same as above, but now with Coordinate3D
  // also, at the end, some convertions are tested
  {
    cerr << "Testing Coordinate3D" << endl;

    Coordinate3D<float> a;
    a[1]=1;a[2]=2;a[3]=3;
    // use new constructor
    Coordinate3D<float> copy_of_a(1,2,3);
    Coordinate3D<float> b;
    b[1]=-1;b[2]=-3;b[3]=5;
    Coordinate3D<float> a_plus_b;
    a_plus_b[1]=0;a_plus_b[2]=-1;a_plus_b[3]=8;
    
    check(a[3]==3, "testing operator[]");
    check_if_equal(inner_product(a,b), 8., "testing inner_product");
    check_if_equal(norm(a), 3.74166, "testing norm");

    a += b;
    check_if_zero(norm(a - a_plus_b), "testing operator+=(BasicCoordinate)");
    a -= b;
    check_if_zero(norm(a - copy_of_a), "testing operator-=(BasicCoordinate)");
    
    {
      Coordinate3D<float> b1 = b;
      check_if_zero(norm(b1-b), "testing copy constructor, and operator-");
      
      b1 = a;
      check_if_zero(norm(a-b1), "testing assignment");
    }

    a *= 4;
    check_if_zero(norm(a)- norm(copy_of_a)*4, "testing operator*=(float)");
    check_if_zero(a[1]- copy_of_a[1]*4, "testing operator*=(float)");
    a /= 4;
    check_if_zero(norm(a-copy_of_a), "testing operator/=(float)");    

    {
      Coordinate3D<float> a1;
      a1 = b;
      a1 *= 3;
      a1 += a;
      a1 -= 4;
      Coordinate3D<float> a2 = (b*3+a)-4;
      check_if_zero(norm(a1-a2), "testing various numerical operators");    
    }

    {
      BasicCoordinate<3,float> gen_a(a);
      a = gen_a;
      check_if_zero(norm(a-copy_of_a), "testing convertions");    
      check_if_zero(norm(gen_a-copy_of_a), "testing convertions");    
    }

  }

  // essentially the same as above, but now with CartesianCoordinate3D
  {
    cerr << "Testing CartesianCoordinate3D" << endl;

    CartesianCoordinate3D<float> a;
    a[1]=1;a[2]=2;a[3]=3;
    // use new constructor
    CartesianCoordinate3D<float> copy_of_a(1,2,3);
    CartesianCoordinate3D<float> b;
    b[1]=-1;b[2]=-3;b[3]=5;
    CartesianCoordinate3D<float> a_plus_b;
    a_plus_b[1]=0;a_plus_b[2]=-1;a_plus_b[3]=8;
    
    check(a[3]==3, "testing operator[]");
    check_if_equal(inner_product(a,b), 8., "testing inner_product");
    check_if_equal(norm(a), 3.74166, "testing norm");

    a += b;
    check_if_zero(norm(a - a_plus_b), "testing operator+=(BasicCoordinate)");
    a -= b;
    check_if_zero(norm(a - copy_of_a), "testing operator-=(BasicCoordinate)");
    
    {
      CartesianCoordinate3D<float> b1 = b;
      check_if_zero(norm(b1-b), "testing copy constructor, and operator-");
      
      b1 = a;
      check_if_zero(norm(a-b1), "testing assignment");
    }

    a *= 4;
    check_if_zero(norm(a)- norm(copy_of_a)*4, "testing operator*=(float)");
    check_if_zero(a[1]- copy_of_a[1]*4, "testing operator*=(float)");
    a /= 4;
    check_if_zero(norm(a-copy_of_a), "testing operator/=(float)");    

    {
      CartesianCoordinate3D<float> a1;
      a1 = b;
      a1 *= 3;
      a1 += a;
      a1 -= 4;
      CartesianCoordinate3D<float> a2 = (b*3+a)-4;
      check_if_zero(norm(a1-a2), "testing various numerical operators");    
    }

    {
      BasicCoordinate<3,float> gen_a(a);
      a = gen_a;
      check_if_zero(norm(a-copy_of_a), "testing convertions");    
      check_if_zero(norm(gen_a-copy_of_a), "testing convertions");    
    }

  }

}


END_NAMESPACE_TOMO



USING_NAMESPACE_TOMO



int main()
{
  coordinateTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
