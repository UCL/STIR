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
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef NDEBUG
// set to high level of debugging
#ifdef _DEBUG
#undef _DEBUG
#endif
#define _DEBUG 2
#endif

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
using std::size_t;
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
  check_if_equal(v.get_min_index(), -3, "test basic constructor and get_min_index");
  check_if_equal(v.get_max_index(), 40, "test basic constructor and get_max_index");
  check(v.size()== 40+3+1, "test basic constructor and size");
  check(v.capacity()== 40+3+1, "test basic constructor and capacity");

  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = 2*i;
  check_if_equal(v[4], 8, "test operator[]");

  int *ptr = v.get_data_ptr();
  ptr[4+3] = 5;
  v.release_data_ptr();
  check_if_equal(v[4], 5, "test get_data_tr/release_data_ptr");

  // iterator tests
  {
    { 
      int value = -3;
      for (VectorWithOffset<int>::iterator iter = v.begin();
	   iter != v.end();
	   iter++, value++)
	*iter = value;
      check_if_equal(v[4], 4, "test iterators operator++ and *");
    }
    {
      VectorWithOffset<int>::const_iterator iter = v.begin();
      check_if_equal(*(iter+4-v.get_min_index()), 4, "test iterators operator+ and *");
      iter += 4-v.get_min_index();
      check_if_equal(*iter, 4, "test iterators operator+= and *");
      ++iter;
      check_if_equal(*(iter-1), 4, "test iterators operator+= and *");
      --iter;
      check_if_equal(*iter, 4, "test iterators operator+= and *");
    }
    {
      const VectorWithOffset<int>& const_v = v;
      VectorWithOffset<int>::const_iterator iter = const_v.begin();
      check_if_equal(*(iter+4-v.get_min_index()), 4, "test iterators operator+ and *");
      iter += 4-v.get_min_index();
      check_if_equal(*iter, 4, "test iterators operator+= and *");
      ++iter;
      check_if_equal(*(iter-1), 4, "test iterators operator+= and *");
      --iter;
      check_if_equal(*iter, 4, "test iterators operator+= and *");
    }


    {
      VectorWithOffset<int>::iterator p=find(v.begin(), v.end(), 6);
      check_if_equal(p - v.begin(), 9, "test iterators: find");
      check_if_equal(*p, 6, "test iterators: find");
    }

    sort(v.begin(), v.end(), greater<int>());
    check_if_equal(v[-3], 40, "test iterators: sort");
    check_if_equal(v[0], 37, "test iterators: sort");

    {
      VectorWithOffset<int>::reverse_iterator pr = v.rbegin();
      check_if_equal(*pr++, -3, "test reverse iterator operator++ and *");
      check_if_equal(*(pr+2), 0, "test reverse iterator operator++ and *");
    }

    sort(v.rbegin(), v.rend(), greater<int>());
    check_if_equal(v[-3], -3, "test reverse iterators after sort");
    check_if_equal(v[0], 0, "test reverse iterators after sort");

  } // end test iterators


  {
    VectorWithOffset<int> test = v;
    test.set_min_index(-1);
    check( test.size()==v.size(), "test set_min_index (size)");
    check( test.capacity()==v.capacity(), "test set_min_index (capacity)");
    check_if_equal( test[-1], v[v.get_min_index()], "test set_min_index (operator[])");
  }

  /**********************************************************************/
  // tests on reserve()
  /**********************************************************************/
  {
    // tests on reserve() with 0 length
    {
      // check reserve 0,max
      {
	VectorWithOffset<int> test;
	check_if_equal(test.capacity(), size_t(0),
		       "check capacity after default constructor");
	test.reserve(2);
	check_if_equal(test.size(), size_t(0),
		       "check reserve of empty vector (0,max) (size)");
	check_if_equal(test.capacity(), size_t(2), 
		       "check reserve of empty vector (0,max) (capacity)");
	check_if_equal(test.get_capacity_min_index(), 0,
		       "check reserve of empty vector (0,max) (capacity_min_index)");
	check_if_equal(test.get_capacity_max_index(), 1,
		       "check reserve of empty vector (0,max) (capacity_max_index)");
      }
      // check reserve -1,2
      {
	VectorWithOffset<int> test;
	test.reserve(-1,2);
	check_if_equal(test.size(), size_t(0),
		       "check reserve of empty vector (-1,2) (size)");
	check_if_equal(test.capacity(), size_t(4), 
		       "check reserve of empty vector (-1,2) (capacity)");
	// note: for length 0 vectors, get_capacity_min_index() is always 0
	check_if_equal(test.get_capacity_min_index(), 0,
		       "check reserve of empty vector (-1,2) (capacity_min_index)");
	check_if_equal(test.get_capacity_max_index(), 3,
		       "check reserve of empty vector (-1,2) (capacity_max_index)");
      }
      // check reserve -1,2 and then 1,6
      {
	VectorWithOffset<int> test;
	test.reserve(-1,2);
	test.reserve(1,6);
	check_if_equal(test.size(), size_t(0),
		       "check reserve of empty vector (-1,2 and then 1,6) (size)");
	check_if_equal(test.capacity(), size_t(6), 
		       "check reserve of empty vector (-1,2 and then 1,6) (capacity)");
	// note: for length 0 vectors, get_capacity_min_index() is always 0
	check_if_equal(test.get_capacity_min_index(), 0,
		       "check reserve of empty vector (-1,2 and then 1,6) (capacity_min_index)");
	check_if_equal(test.get_capacity_max_index(), 5,
		       "check reserve of empty vector (-1,2 and then 1,6) (capacity_max_index)");
      }
    } // end of tests length 0

    // tests of reserve() with non-zero length
    {
      const VectorWithOffset<int> ref = v;
      VectorWithOffset<int> test = ref;
      // check reserve within range (should have no effect)
      test.reserve(0,1);
      check_if_equal(test.size(), ref.size(), 
		     "check reserve within range (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check reserve within range (get_min_index)");
      check_if_equal(test, ref, 
		     "check reserve within range (values)");
      check_if_equal(test.capacity(), ref.size(), 
		     "check reserve within range (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check reserve within range (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index(), 
		     "check reserve within range (capacity_max_index)");
      // check reserve within range on low index (should reserve space at higher indices only)
      test.reserve(0,test.get_max_index()+5);
      check_if_equal(test.size(), ref.size(), 
		     "check reserve within range on low index (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check reserve within range on low index (get_min_index)");
      check_if_equal(test, ref, 
		     "check reserve within range on low index (values)");
      check_if_equal(test.capacity(), ref.size()+5, 
		     "check reserve within range on low index (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check reserve within range on low index (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index()+5, 
		     "check reserve within range on low index (capacity_max_index)");
      // check reserve within range on high index (should reserve space at low indices only)
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.reserve(test.get_min_index()-5,0);
      check_if_equal(test.size(), ref.size(), 
		     "check reserve within range on high index (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check reserve within range on high index (get_min_index)");
      check_if_equal(test, ref, 
		     "check reserve within range on high index (values)");
      check_if_equal(test.capacity(), ref.size()+5, 
		     "check reserve within range on high index (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index()-5, 
		     "check reserve within range on high index (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index(), 
		     "check reserve within range on high index (capacity_max_index)");
      // check reserve for both ranges
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.reserve(test.get_min_index()-5,test.get_max_index()+4);
      check_if_equal(test.size(), ref.size(), 
		     "check reserve (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check reserve (get_min_index)");
      check_if_equal(test, ref, 
		     "check reserve (values)");
      check_if_equal(test.capacity(), ref.size()+9, 
		     "check reserve (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index()-5, 
		     "check reserve (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index()+4, 
		     "check reserve (capacity_max_index)");
    }
  }  
  /**********************************************************************/
  // tests on resize()
  /**********************************************************************/
  {
    // tests on resize() with 0 length
    {
      // check resize 0,max
      {
	VectorWithOffset<int> test;
	check_if_equal(test.capacity(), size_t(0),
		       "check capacity after default constructor");
	test.resize(2);
	check_if_equal(test.size(), size_t(2),
		       "check resize of empty vector (0,max) (size)");
	check_if_equal(test.capacity(), size_t(2), 
		       "check resize of empty vector (0,max) (capacity)");
	check_if_equal(test.get_min_index(), 0,
		       "check resize of empty vector (0,max) (min_index)");
	check_if_equal(test.get_max_index(), 1,
		       "check resize of empty vector (0,max) (max_index)");
      }
      // check resize -1,2
      {
	VectorWithOffset<int> test;
	test.resize(-1,2);
	check_if_equal(test.size(), size_t(4),
		       "check resize of empty vector (-1,2) (size)");
	check_if_equal(test.capacity(), size_t(4), 
		       "check resize of empty vector (-1,2) (capacity)");
	check_if_equal(test.get_min_index(), -1,
		       "check resize of empty vector (-1,2) (min_index)");
	check_if_equal(test.get_max_index(), 2,
		       "check resize of empty vector (-1,2) (max_index)");
      }
      // check resize -1,2 and then 1,6
      {
	VectorWithOffset<int> test;
	test.resize(-1,2);
	test.resize(1,6);
	check_if_equal(test.size(), size_t(6),
		       "check resize of empty vector (-1,2 and then 1,6) (size)");
	check_if_equal(test.capacity(), size_t(8), 
		       "check resize of empty vector (-1,2 and then 1,6) (capacity)");
	// note: for length 0 vectors, get_min_index() is always 0
	check_if_equal(test.get_min_index(), 1,
		       "check resize of empty vector (-1,2 and then 1,6) (min_index)");
	check_if_equal(test.get_max_index(), 6,
		       "check resize of empty vector (-1,2 and then 1,6) (max_index)");
      }
    } // end of tests length 0

    // tests of resize() with non-zero length
    {
      const VectorWithOffset<int> ref = v;
      VectorWithOffset<int> test = ref;
      // check resize with identical range (should have no effect)
      test.resize(ref.get_min_index(), ref.get_max_index());
      check_if_equal(test.size(), ref.size(), 
		     "check resize with identical range (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check resize with identical range (get_min_index)");
      check_if_equal(test, ref, 
		     "check resize with identical range (values)");
      check_if_equal(test.capacity(), ref.size(), 
		     "check resize with identical range (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check resize with identical range (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index(), 
		     "check resize with identical range (capacity_max_index)");
      // check resize with grow on high index (should resize space at higher indices only)
      test.resize(ref.get_min_index(),test.get_max_index()+5);
      check_if_equal(test.size(), ref.size()+5, 
		     "check resize with grow on high index (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index(), 
		     "check resize with grow on high index (get_min_index)");
      for (int i=ref.get_min_index(); i<=ref.get_max_index(); ++i)
	check_if_equal(test[i], ref[i], 
		       "check resize with grow on high index (values)");
      check_if_equal(test.capacity(), ref.size()+5, 
		     "check resize with grow on high index (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check resize with grow on high index (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index()+5, 
		     "check resize with grow on high index (capacity_max_index)");
      // check resize with grow on low index (should resize space at low indices only)
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.resize(test.get_min_index()-5,ref.get_max_index());
      check_if_equal(test.size(), ref.size()+5, 
		     "check resize with grow on low index (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index()-5, 
		     "check resize with grow on low index (get_min_index)");
      for (int i=ref.get_min_index(); i<=ref.get_max_index(); ++i)
	check_if_equal(test[i], ref[i], 
		       "check resize with grow on low index (values)");
      check_if_equal(test.capacity(), ref.size()+5, 
		     "check resize with grow on low index (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index()-5, 
		     "check resize with grow on low index (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index(), 
		     "check resize with grow on low index (capacity_max_index)");
      // check grow for both ranges
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.resize(test.get_min_index()-5,test.get_max_index()+4);
      check_if_equal(test.size(), ref.size()+9, 
		     "check resize with grow at both ends (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index()-5, 
		     "check resize with grow at both ends (get_min_index)");
      for (int i=ref.get_min_index(); i<=ref.get_max_index(); ++i)
	check_if_equal(test[i], ref[i], 
		       "check resize with grow at both ends (values)");
      check_if_equal(test.capacity(), ref.size()+9, 
		     "check resize with grow at both ends (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index()-5, 
		     "check resize with grow at both ends (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index()+4, 
		     "check resize with grow at both ends (capacity_max_index)");
      // check resize with shrink for both ranges
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.resize(test.get_min_index()+5,test.get_max_index()-4);
      check_if_equal(test.size(), ref.size()-9, 
		     "check resize with shrink at both ends(size)");
      check_if_equal(test.get_min_index(), ref.get_min_index()+5, 
		     "check resize with shrink at both ends(get_min_index)");
      for (int i=test.get_min_index(); i<=test.get_max_index(); ++i)
	check_if_equal(test[i], ref[i], 
		       "check resize with shrink at both ends (values)");
      check_if_equal(test.capacity(), ref.size(), 
		     "check resize with shrink at both ends(capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check resize with shrink at both ends(capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index(), 
		     "check resize with shrink at both ends(capacity_max_index)");
      // check resize with shrink at left and grow at right
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      test.resize(test.get_min_index()+5,test.get_max_index()+4);
      check_if_equal(test.size(), ref.size()-1, 
		     "check resize with shrink at left and grow at right (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index()+5, 
		     "check resize with shrink at left and grow at right (get_min_index)");
      for (int i=test.get_min_index(); i<=ref.get_max_index(); ++i)
	check_if_equal(test[i], ref[i], 
		       "check resize with shrink at left and grow at right (values)");
      check_if_equal(test.capacity(), ref.size()+4, 
		     "check resize with shrink at left and grow at right (capacity)");
      check_if_equal(test.get_capacity_min_index(), ref.get_min_index(), 
		     "check resize with shrink at left and grow at right (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), ref.get_max_index()+4, 
		     "check resize with shrink at left and grow at right (capacity_max_index)");
      // check resize with resize non-overlapping to right
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      // not: test new size is smaller than original size, so no reallocation should occur
      test.resize(test.get_max_index()+20,test.get_max_index()+20+ref.size()-3);
      check_if_equal(test.size(), size_t(ref.size()-2), 
		     "check resize with resize non-overlapping to right (size)");
      check_if_equal(test.get_min_index(), ref.get_max_index()+20, 
		     "check resize with resize non-overlapping to right (get_min_index)");
      check_if_equal(test.capacity(), ref.size(), 
		     "check resize with resize non-overlapping to right (capacity)");
      check_if_equal(test.get_capacity_min_index(), test.get_min_index(), 
		     "check resize with resize non-overlapping to right (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), test.get_min_index()+static_cast<int>(test.capacity())-1, 
		     "check resize with resize non-overlapping to right (capacity_max_index)");
      // check resize with resize non-overlapping to left
      test.recycle();
      check_if_equal(test.capacity(), size_t(0), "test recycle");
      test = ref;
      // not: test new size is larger than original size, so reallocation should occur
      test.resize(test.get_min_index()-300,test.get_min_index()-300+ref.size()+4);
      check_if_equal(test.size(), size_t(ref.size()+5), 
		     "check resize with resize non-overlapping to left (size)");
      check_if_equal(test.get_min_index(), ref.get_min_index()-300, 
		     "check resize with resize non-overlapping to left (get_min_index)");
      check_if_equal(test.capacity(), test.size(), 
		     "check resize with resize non-overlapping to left (capacity)");
      check_if_equal(test.get_capacity_min_index(), test.get_min_index(), 
		     "check resize with resize non-overlapping to left (capacity_min_index)");
      check_if_equal(test.get_capacity_max_index(), test.get_max_index(), 
		     "check resize with resize non-overlapping to left (capacity_max_index)");
    }
  }  

  /**********************************************************************/
  // tests on operator += etc
  /**********************************************************************/
  {
    const VectorWithOffset<int> ref = v;
    VectorWithOffset<int> test = ref;

    test = ref; test += ref;
    for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
      check_if_equal( test[i] , ref[i]*2, "test operator+=(VectorWithOffset)");
    test = ref; test -= ref;
    for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
      check_if_equal( test[i] , 0, "test operator-=(VectorWithOffset)");
    test = ref; test *= ref;
    for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
      check_if_equal( test[i] , ref[i]*ref[i], "test operator*=(VectorWithOffset)");
    
    const int minimum = *std::min_element(ref.begin(), ref.end());
    const int ensure_non_zero = 
       minimum<= 0 ? -minimum+1 : 0;
    VectorWithOffset<int> denominator = ref; 
    test = ref;
    for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
      {
	denominator[i] += ensure_non_zero;
	test[i]+=10000;
      }
    test /= denominator;
    for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
      check_if_equal( test[i] , (ref[i]+10000)/(ref[i]+ensure_non_zero), "test operator/=(VectorWithOffset)");

  }

  // some checks with min_index==0
  {
    VectorWithOffset<int> v0(40);
    check_if_equal(v0.get_min_index(), 0, "test 1-arg constructor and get_min_index");
    check_if_equal(v0.get_max_index(), 40-1, "test 1-arg constructor and get_max_index");
    check_if_equal(v0.size(), size_t(40), "test 1-arg constructor and size");
    check_if_equal(v0.capacity(), size_t(40), "test 1-arg constructor and capacity");
  }
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  VectorWithOffsetTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
