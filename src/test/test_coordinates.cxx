//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
 
  \brief A simple program to test the Coordinate classes

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/

#include "stir/Coordinate2D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Coordinate4D.h"
#include "stir/round.h"
#include "stir/RunTests.h"
#include <iostream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

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
#ifndef STIR_NO_NAMESPACES
      float *p=std::find(b.begin(), b.end(), -3);
#else
      float *p=find(b.begin(), b.end(), -3);
#endif
      check_if_equal(p - b.begin(), 1, "iterator test");
      BasicCoordinate<3, float> b_sorted;
      b_sorted[1]=-3;b_sorted[2]=-1;b_sorted[3]=5;
#ifndef STIR_NO_NAMESPACES
      std::sort(b.begin(), b.end()); 
#else
      sort(b.begin(), b.end()); 
#endif
      check_if_zero(norm(b-b_sorted), "testing iterators via STL sort");
    }
  }
  {
    cerr << "Testing join/cut_first_dimension/comparisons on BasicCoordinate<?,int>" << endl;
    
    // join
    {
      BasicCoordinate<3, int> a;
      a[1]=1;a[2]=2;a[3]=3;
      {
	BasicCoordinate<4, int> a4 = join(0, a);
	check_if_equal(a4[1], 0, "testing join of float with BasicCoordinate");
	check_if_equal(a4[2], 1, "testing join of float with BasicCoordinate");
	check_if_equal(a4[3], 2, "testing join of float with BasicCoordinate");
	check_if_equal(a4[4], 3, "testing join of float with BasicCoordinate");
      }
      {
	BasicCoordinate<4, int> a4 = join(a, 0);
	check_if_equal(a4[1], 1, "testing join of BasicCoordinate with float");
	check_if_equal(a4[2], 2, "testing join of BasicCoordinate with float");
	check_if_equal(a4[3], 3, "testing join of BasicCoordinate with float");
	check_if_equal(a4[4], 0, "testing join of BasicCoordinate with float");
      }
    }
    // cut*dimension
    {
      BasicCoordinate<3, int> a;
      a[1]=1;a[2]=2;a[3]=3;
      const BasicCoordinate<2, int> start = cut_last_dimension(a);
      check_if_equal(start[1], 1, "testing cut_last_dimension");
      check_if_equal(start[2], 2, "testing cut_last_dimension");
      const BasicCoordinate<2, int> end = cut_first_dimension(a);
      check_if_equal(end[1], 2, "testing cut_first_dimension");
      check_if_equal(end[2], 3, "testing cut_first_dimension");
    }
    // comparison 2D
    {
      BasicCoordinate<2, int> a;
      a[1]=1;a[2]=2;
      BasicCoordinate<2, int> b;
      b[1]=1;b[2]=1;
      check(a==a, "2D operator==");
      check(a<=a, "2D operator<= (when equal)");
      check(a>=a, "2D operator>= (when equal)");
      check(a>b, "2D operator>");
      check(b<a, "2D operator<");
      check(a>=b, "2D operator>= (when not equal)");
      check(b<=a, "2D operator<= (when not equal)");
      check(a!=b, "2D operator!=");
    }
    // comparison 3D
    {
      BasicCoordinate<3, int> a;
      a[1]=1;a[2]=2;a[3]=3;
      BasicCoordinate<3, int> b;
      b[1]=1;b[2]=1;b[3]=3;
      check(a==a, "3D operator==");
      check(a<=a, "3D operator<= (when equal)");
      check(a>=a, "3D operator>= (when equal)");
      check(a>b, "3D operator>");
      check(b<a, "3D operator<");
      check(a>=b, "3D operator>= (when not equal)");
      check(b<=a, "3D operator<= (when not equal)");
      check(a!=b, "3D operator!=");
    }
    // comparison 4D
    {
      BasicCoordinate<4, int> a;
      a[1]=1;a[2]=2;a[3]=3; a[4]=1;
      BasicCoordinate<4, int> b;
      b[1]=1;b[2]=1;b[3]=3; b[4]=2;
      check(a==a, "4D operator==");
      check(a<=a, "4D operator<= (when equal)");
      check(a>=a, "4D operator>= (when equal)");
      check(a>b, "4D operator>");
      check(b<a, "4D operator<");
      check(a>=b, "4D operator>= (when not equal)");
      check(b<=a, "4D operator<= (when not equal)");
      check(a!=b, "4D operator!=");
    }
  }

  // essentially the same as BasicCoordinate<3,float>, but now with Coordinate3D
  // also, at the end, some conversions are tested
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
      check_if_zero(norm(a-copy_of_a), "testing conversions");    
      check_if_zero(norm(gen_a-copy_of_a), "testing conversions");    
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
      check_if_zero(norm(a-copy_of_a), "testing conversions");    
      check_if_zero(norm(gen_a-copy_of_a), "testing conversions");    
    }

  }

  {
    cerr << "Testing round with coordinates" << endl;
    const Coordinate3D<float> af(1.1F,-1.1F,3.6F);
    const Coordinate3D<int> aint = round(af);
    check_if_equal(aint, Coordinate3D<int> (1,-1,4));
  }
  {
    cerr << "Testing constructor with different types of coordinates" << endl;
    /* Note: C++ rules for automatic conversion are such that the code below
       would not work if 'af' is defined as
       const Coordinate3D<float> af(1.1F,-1.1F,3.6F);
    */
    const BasicCoordinate<3,float> af = Coordinate3D<float> (1.1F,-1.1F,3.6F);
    const BasicCoordinate<3,int> aint(af);
    check_if_equal(aint, Coordinate3D<int> (1,-1,3));
    const BasicCoordinate<3,float> af2(aint);
    check_if_equal(af2, Coordinate3D<float> (1.F,-1.F,3.F));
  }

  {
    cerr << "Testing make_coordinate" << endl;
    check_if_equal(make_coordinate(1.F)[1],1.F, "test make_coordinate with 1 arg");
    check_if_equal(make_coordinate(1.F,3.4F),Coordinate2D<float>(1.F,3.4F), 
		   "test make_coordinate with 2 args");
    check_if_equal(make_coordinate(1.,3.4,-4.8),Coordinate3D<double>(1.,3.4,-4.8), 
		   "test make_coordinate with 3 args");
    check_if_equal(make_coordinate(1,2,3,4),Coordinate4D<int>(1,2,3,4), 
		   "test make_coordinate with 4 args");
    check_if_equal(make_coordinate(1,2,3,4,5),join(Coordinate4D<int>(1,2,3,4),5), 
		   "test make_coordinate with 5 args");
    check_if_equal(make_coordinate(1,2,3,4,5,6),join(join(Coordinate4D<int>(1,2,3,4),5),6), 
		   "test make_coordinate with 6 args");
  }
}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR



int main()
{
  coordinateTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
