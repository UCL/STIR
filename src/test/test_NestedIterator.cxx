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
#include "stir/shared_ptr.h"
#include <iostream>
#include <list>
#include <vector>

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
# if defined __GNUC__ &&  __GNUC__ < 3
    // at some point, it seemed we needed a work-around for gcc 3 or earlier, but that is no longer the case
    typedef  BeginEndFunction<Array<2,elemT>::const_iterator> constbeginendfunction_type;
#else
    typedef ConstBeginEndFunction<Array<2,elemT>::const_iterator> constbeginendfunction_type;
#endif

    check(test2.begin()->begin() == constbeginendfunction_type().begin(test2.begin()), "begin");
    check((test2.begin()+1)->begin() == constbeginendfunction_type().begin(test2.begin()+1), "begin");
    typedef NestedIterator<Array<2,elemT>::const_iterator, constbeginendfunction_type>  
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
  std::cerr<< " NestedIterator vector<list>\n";
  {
    typedef std::list<int> C2;
    typedef std::vector<C2> C1;
    C1 c(2);
    c[0].push_back(1); c[0].push_back(2);
    c[1].push_back(3); c[1].push_back(4); c[1].push_back(5);
    // normal iterator
    {
      typedef NestedIterator<C1::iterator> FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      int count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector<list>");
      }
      check_if_equal(count, 6, "nestiterator of vector<list>: num elements");
    }
    // const iterator
    {
      typedef NestedIterator<C1::const_iterator, ConstBeginEndFunction<C1::const_iterator> > FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      int count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector<list>");
      }
      check_if_equal(count, 6, "nestiterator of vector<list>: num elements");
    }
    // test conversion from full_iterator to const_full_iterator
    {
      typedef NestedIterator<C1::const_iterator, ConstBeginEndFunction<C1::const_iterator> > CFullIter;
      typedef NestedIterator<C1::iterator> FullIter;
      FullIter fiter(c.begin(),c.end());
      CFullIter cfiter= fiter; // this should compile
    }
  }
  std::cerr<< " NestedIterator vector<list *>\n";
  {
    typedef std::list<int> C2;
    typedef std::vector<C2 *> C1;
    C1 c(2);
    c[0] = new C2;
    c[1] = new C2;
    c[0]->push_back(1); c[0]->push_back(2);
    c[1]->push_back(3); c[1]->push_back(4); c[1]->push_back(5);
    // normal iterator
    {
      typedef NestedIterator<C1::iterator, PtrBeginEndFunction<C1::iterator> > FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      int count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector< list *>");
      }
      check_if_equal(count, 6, "nestiterator of vector<list *>: num elements");
    }
    // const iterator
    {
      typedef NestedIterator<C1::const_iterator, ConstPtrBeginEndFunction<C1::const_iterator> > FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      int count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector< list *>");
      }
      check_if_equal(count, 6, "nestiterator of vector<list *>: num elements");
    }
    delete c[0];
    delete c[1];
  }

  std::cerr<< " NestedIterator vector< shared_ptr<Array<2,...> >\n";
  {
    IndexRange<2> range1(make_coordinate(0,0),make_coordinate(1,2));
    IndexRange<2> range2(make_coordinate(0,0),make_coordinate(1,3));
    typedef int elemT;
    typedef Array<2,int> C2;
    typedef std::vector<shared_ptr<C2> > C1;
    C1 c(2);
    c[0] = new C2(range1);
    c[1] = new C2(range2);
    int count = 1;
    for (C2::full_iterator fullarrayiter = c[0]->begin_all(); fullarrayiter != c[0]->end_all();
         ++fullarrayiter, ++count)
      *fullarrayiter = count;
    for (C2::full_iterator fullarrayiter = c[1]->begin_all(); fullarrayiter != c[1]->end_all();
         ++fullarrayiter, ++count)
      *fullarrayiter = count;

    // normal iterator
    {
      typedef NestedIterator<C1::iterator, PtrBeginEndAllFunction<C1::iterator> > FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector<shared_ptr<Array<2,int>>>");
      }
      check_if_equal(count, static_cast<int>(c[0]->size_all() + c[1]->size_all() + 1), 
        "nestiterator of vector<shared_ptr<Array<2,int>>>: num elements");
    }
    // const iterator
    {
      typedef NestedIterator<C1::const_iterator, ConstPtrBeginEndAllFunction<C1::const_iterator> > FullIter;
      FullIter fiter(c.begin(),c.end());
      const FullIter fiter_end(c.end(),c.end());
      count=1;
      while (fiter != fiter_end)
      { 
        check_if_equal(*fiter++, count++, "nestiterator of vector<shared_ptr<Array<2,int>>>");
      }
      check_if_equal(count, static_cast<int>(c[0]->size_all() + c[1]->size_all() + 1), 
        "nestiterator of vector<shared_ptr<Array<2,int>>>: num elements");
    }
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
