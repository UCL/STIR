// $Id$
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
 
  \brief tests for the stir::NestedIterator class

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/Succeeded.h"

#include "stir/RunTests.h"
#include "stir/NestedIterator.h"
#include <iostream>

START_NAMESPACE_STIR

/*!
  \brief Tests NestedIterator functionality
  \ingroup test

  \todo Code is ugly. Copy-paste with tiny modifications.
*/
class NestedIteratorTests : public RunTests
{
private:

public:
  void run_tests();
};


void
NestedIteratorTests::run_tests()
{

  std::cerr << "Testing NestedIterator\n";
  {
    std::cerr << "Compare with full iterator\n";
    {
      IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
      Array<2,float> test2(range);

      {
	float value = 1.2F;
	for (Array<2,float>::full_iterator iter = test2.begin_all();
	     iter != test2.end_all(); 
	     )
	  *iter++ = value++;
      }

      check(test2.begin()->begin() == BeginEndFunction<Array<2,float>::iterator>().begin(test2.begin()), "begin");
      check((test2.begin()+1)->begin() == BeginEndFunction<Array<2,float>::iterator>().begin(test2.begin()+1), "begin");
      typedef NestedIterator<Array<2,float>::iterator> FullIter;
      FullIter fiter1;
      FullIter fiter(test2.begin(),test2.end());
      //fiter1=fiter;
      const FullIter fiter_end(test2.end(),test2.end());      
      for (Array<2,float>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check(*fiter++ == *iter++,"fiter==");
	}
      
      check(fiter==fiter_end,"fiter end");      
      //      check(test2.begin()->begin() == make_begin_function(test2.begin())(), "begin");

      const Array<2,float> empty;
      check(empty.begin_all() == empty.end_all(), "test on 2D full iterator for empty range");
    }
  }

  std::cerr<< " full iterator coord\n";
  {
    IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
    typedef BasicCoordinate<2,float> elemT;
    Array<2,elemT > test2(range);

    {
      float value = 1.2F;
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  *iter++ = make_coordinate(value,value+.3F);
	  ++value;
	}
    }

    check(test2.begin()->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()), "begin");
    check((test2.begin()+1)->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()+1), "begin");
    typedef NestedIterator<Array<2,elemT>::iterator>	
      FullIter;
    FullIter fiter1;
    FullIter fiter(test2.begin(),test2.end());
    //fiter1=fiter;
    const FullIter fiter_end(test2.end(),test2.end());      
    {
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check(*fiter++ == *iter++,"fiter==");
	}
    }
    check(fiter==fiter_end,"fiter end");      
  }

  std::cerr<< " full iterator coord\n";
  {
    IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
    typedef BasicCoordinate<2,float> elemT;
    Array<2,elemT > test(range);

    {
      float value = 1.2F;
      for (Array<2,elemT>::full_iterator iter = test.begin_all();
	   iter != test.end_all(); 
	   )
	{
	  *iter++ = make_coordinate(value,value+.3F);
	  ++value;
	}
    }

    const Array<2,elemT> test2 = test;

    check(test2.begin()->begin() == BeginEndFunction<Array<2,elemT>::const_iterator>().begin(test2.begin()), "begin");
    check((test2.begin()+1)->begin() == BeginEndFunction<Array<2,elemT>::const_iterator>().begin(test2.begin()+1), "begin");
    typedef NestedIterator<Array<2,elemT>::const_iterator>	
      FullIter;
    FullIter fiter1;
    FullIter fiter(test2.begin(),test2.end());
    //fiter1=fiter;
    FullIter fiter_end(test2.end(),test2.end());
      
    {
      for (Array<2,elemT>::const_full_iterator iter = test2.begin_all_const();
	   iter != test2.end_all_const(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check(*fiter++ == *iter++,"fiter==");
	}
    }
    check(fiter==fiter_end,"fiter end");      
  }

  std::cerr<< " full iterator coord full\n";
  {
    IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
    typedef BasicCoordinate<2,float> elemT;
    Array<2,elemT > test2(range);

    {
      float value = 1.2F;
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  *iter++ = make_coordinate(value,value+.3F);
	  ++value;
	}
    }

    check(test2.begin()->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()), "begin");
    check((test2.begin()+1)->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()+1), "begin");
    typedef
      NestedIterator<Array<2,elemT>::full_iterator,
      BeginEndFunction<Array<2,elemT>::full_iterator> >
      FullIter;
    FullIter fiter1;
    FullIter fiter(test2.begin_all(),test2.end_all());
    //fiter1=fiter;
    FullIter fiter_end(test2.end_all(),test2.end_all());
      
    {
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check_if_equal(*fiter++, *iter->begin(),"fiter== 0");
	  check_if_equal(*fiter++,*(iter->begin()+1),"fiter== 1");
	  ++iter;
	}
    }
    check(fiter==fiter_end,"fiter end");      
  }

  std::cerr<< " full iterator coord full const\n";
  {
    IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
    typedef BasicCoordinate<2,float> elemT;
    Array<2,elemT > test(range);

    {
      float value = 1.2F;
      for (Array<2,elemT>::full_iterator iter = test.begin_all();
	   iter != test.end_all(); 
	   )
	{
	  *iter++ = make_coordinate(value,value+.3F);
	  ++value;
	}
    }
    const Array<2,elemT> test2 = test;

    check(test2.begin_all_const()->begin() == ConstBeginEndFunction<Array<2,elemT>::const_full_iterator>().begin(test2.begin_all_const()), "begin");
    typedef
      NestedIterator<Array<2,elemT>::const_full_iterator,
      ConstBeginEndFunction<Array<2,elemT>::const_full_iterator> >
      FullIter;
    FullIter fiter1;
    FullIter fiter(test2.begin_all_const(),test2.end_all_const());
    //fiter1=fiter;
    FullIter fiter_end(test2.end_all_const(),test2.end_all_const());
      
    {
      for (Array<2,elemT>::const_full_iterator iter = test2.begin_all_const();
	   iter != test2.end_all_const(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check_if_equal(*fiter++, *iter->begin(),"fiter== 0");
	  check_if_equal(*fiter++,*(iter->begin()+1),"fiter== 1");
	  ++iter;
	}
    }
    check(fiter==fiter_end,"fiter end");      
  }

  std::cerr<< " full iterator coord full\n";
  {
    IndexRange<2> range(make_coordinate(0,0),make_coordinate(2,2));
    typedef BasicCoordinate<2,float> elemT;
    Array<2,elemT > test2(range);

    {
      float value = 1.2F;
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  *iter++ = make_coordinate(value,value+.3F);
	  ++value;
	}
    }

    check(test2.begin()->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()), "begin");
    check((test2.begin()+1)->begin() == BeginEndFunction<Array<2,elemT>::iterator>().begin(test2.begin()+1), "begin");
    typedef
      NestedIterator<Array<2,elemT>::full_iterator,
      BeginEndFunction<Array<2,elemT>::full_iterator> >
      FullIter;
    FullIter fiter1;
    FullIter fiter(test2.begin_all(),test2.end_all());
    //fiter1=fiter;
    FullIter fiter_end(test2.end_all(),test2.end_all());
      
    {
      for (Array<2,elemT>::full_iterator iter = test2.begin_all();
	   iter != test2.end_all(); 
	   )
	{
	  check(fiter!=fiter_end, "fiter");
	  check_if_equal(*fiter++, *iter->begin(),"fiter== 0");
	  check_if_equal(*fiter++,*(iter->begin()+1),"fiter== 1");
	  ++iter;
	}
    }
    check(fiter==fiter_end,"fiter end");      
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  NestedIteratorTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
