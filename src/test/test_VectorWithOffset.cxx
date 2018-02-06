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

  \brief Test program for stir::VectorWithOffset

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
      check_if_zero(p - v.begin() - 9, "test iterators: find");
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
      // note: test new size is smaller than original size, so no reallocation should occur
      test.resize(test.get_max_index()+20, static_cast<int>(test.get_max_index()+20+ref.size()-3));
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
      test.resize(test.get_min_index()-300, static_cast<int>(test.get_min_index()-300+ref.size()+4));
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

  // tests on empty
  {
    {
      VectorWithOffset<int> test;
      check(test.empty(), "test default constructor gives empty vector");
    }
    {
      VectorWithOffset<int> test(1,-1);
      check(test.empty(), "test reverse range gives empty vector");
    }
    {
      VectorWithOffset<int> test(3,6);
      check(!test.empty(), "test vector says !empty()");
      test.resize(0);
      check(test.empty(), "test vector resized to size 0 is empty()");
    }
  }

  // tests on at() with out-of-range
  {
    {
      VectorWithOffset<int> test;
      try
	{
	  int a=test.at(5);
	  // if we get here, there's a problem, so we report that by failing the next test.
	  check(false, "test out-of-range on empty vector");
	}
      catch (std::out_of_range& )
	{
	}
    }
    {
      VectorWithOffset<int> test(1,54);
      try
	{
	  test[4]=1;
	  check_if_equal(test.at(4),1, "test using at() to read content");
	  test.at(3)=2;
	  check_if_equal(test[3],2, "test using at() to set content");

	  int a=test.at(55);
	  // if we get here, there's a problem, so we report that by failing the next test.
	  check(false, "test out-of-range on vector");
	}
      catch (std::out_of_range& )
	{
	}
    }
  }

  // checks on using existing data_ptr with constructor indices starting at 0
  {
    const int size=100;
    int data[size];
    std::fill(data, data+size, 12345);
    check_if_equal(data[0], 12345, "test filling data block at 0");
    check_if_equal(data[size-1], 12345, "test filling data block at end");
    // set data_ptr to somewhere in the block to check overrun
    int * data_ptr = data+10;

    const int vsize = size-20;
    VectorWithOffset<int> v(vsize, data_ptr, data + size);
    check(!v.owns_memory_for_data(), "test vector using data_ptr: should not allocate new memory");
    check(data_ptr == v.get_data_ptr(), "test vector using data_ptr: get_data_ptr()");
    v.release_data_ptr();
    check_if_equal(v[1], 12345, "test vector using data_ptr: vector at 1 after construction");
    v[1]=1;
    check_if_equal(data_ptr[1], 1, "test filling vector using data_ptr: data at 1 after setting");
    check_if_equal(v[1], 1, "test filling vector using data_ptr: vector at 1 after setting");
    v.fill(2);
    check_if_equal(std::accumulate(v.begin(), v.end(), 0), 2*vsize , "test filling vector using data_ptr");
    check_if_equal(data_ptr[0], 2, "test filling vector using data_ptr: data at 0");
    check_if_equal(data_ptr[-1], 12345, "test filling vector using data_ptr: data block before vector");
    check_if_equal(data_ptr[vsize], 12345, "test filling vector using data_ptr: data block after vector");

    // test resize using existing memory
    v[1]=5;
    v.resize(1,vsize-5);
    check(!v.owns_memory_for_data(), "test vector using data_ptr: resize should not allocate new memory");
    check(data_ptr+1 == v.get_data_ptr(), "test vector using data_ptr: get_data_ptr() after resize");
    v.release_data_ptr();
    check_if_equal(v[1], 5 , "test resizing vector using data_ptr: data at 1");
    v[1]=6;
    check_if_equal(data_ptr[1], 6, "test resizing vector using data_ptr: data should still be refered to");
    check_if_equal(std::accumulate(v.begin(), v.end(), 0), 6+v[2]*(v.get_length()-1) , "test resizing vector using data_ptr");

    // test resize that should allocate new memory
    v.resize(-1,vsize-2);
    check(v.owns_memory_for_data(), "test vector using data_ptr: resize should allocate new memory");
    v.fill(7);
    check_if_equal(data[9], 12345, "test vector using data_ptr: after resize data block at 9");
    check_if_equal(data[size-9], 12345, "test vector using data_ptr: after resize data block at end-9");
    check_if_equal(data_ptr[1], 6, "test vector using data_ptr: after resize data 1");
  }

  // checks on using existing data_ptr with constructor indices starting at -3
  {
    const int size=100;
    int data[size];
    std::fill(data, data+size, 12345);
    check_if_equal(data[0], 12345, "test filling data block at 0");
    check_if_equal(data[size-1], 12345, "test filling data block at end");
    // set data_ptr to somewhere in the block to check overrun
    int * data_ptr = data+10;

    const int vsize = size-20;
    VectorWithOffset<int> v(-3, vsize-4, data_ptr, data + size);
    check_if_equal(v.get_length(), vsize, "test vector using data_ptr (negative min_index):size");
    // first essentially same tests as above
    check(!v.owns_memory_for_data(), "test vector using data_ptr (negative min_index): should not allocate new memory");
    check(data_ptr == v.get_data_ptr(), "test vector using data_ptr (negative min_index): get_data_ptr()");
    v.release_data_ptr();
    check_if_equal(v[1], 12345, "test vector using data_ptr (negative min_index): vector at 1 after construction");
    check_if_equal(v[-3], 12345, "test vector using data_ptr (negative min_index): vector at -3 after construction");
    v[-3]=1;
    check_if_equal(data_ptr[0], 1, "test filling vector using data_ptr (negative min_index) data at -3 after setting");
    check_if_equal(v[-3], 1, "test filling vector using data_ptr (negative min_index) vector at -3 after setting");
    v.fill(2);
    check_if_equal(std::accumulate(v.begin(), v.end(), 0), 2*vsize , "test filling vector using data_ptr (negative min_index)");
    check_if_equal(data_ptr[0], 2, "test filling vector using data_ptr (negative min_index) data at 0");
    check_if_equal(data_ptr[-1], 12345, "test filling vector using data_ptr (negative min_index) data block before vector");
    check_if_equal(data_ptr[vsize], 12345, "test filling vector using data_ptr (negative min_index) data block after vector");

    // assignment that doesn't reallocate
    v = VectorWithOffset<int>(2,6);
    check(!v.owns_memory_for_data(), "test vector using data_ptr (negative min_index): assignment should not allocate new memory");
    v[4]=4;
    check_if_equal(v[4], 4, "test vector using data_ptr (negative min_index): vector at 4 after assignment and setting");
    // vector will again start at data_ptr
    check_if_equal(data_ptr[4-v.get_min_index()], 4, "test vector using data_ptr (negative min_index): data at 4-min_index after assignment and setting");
    // another assignment that does not reallocate
    v = VectorWithOffset<int>(static_cast<int>(size-(data_ptr-data) ));
    check(!v.owns_memory_for_data(), "test vector using data_ptr (negative min_index): 2nd assignment should not allocate new memory");
    // assignment that does reallocate
    v = VectorWithOffset<int>(static_cast<int>(size-(data_ptr-data)+1 ));
    check(v.owns_memory_for_data(), "test vector using data_ptr (negative min_index): 3rd assignment should allocate new memory");     
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
