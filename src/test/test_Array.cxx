/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2011, Hammersmith Imanet Ltd
    Copyright (C) 2013 Kris Thielemans
    Copyright (C) 2013 University College London

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
  \ingroup Array
 
  \brief tests for the stir::Array class

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

#include "stir/Array.h"
#include "stir/Coordinate2D.h"
#include "stir/Coordinate3D.h"
#include "stir/Coordinate4D.h"
#include "stir/convert_array.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"

#include "stir/RunTests.h"

#include "stir/ArrayFunction.h"
#include "stir/array_index_functions.h"
#include <functional>

// for open_read/write_binary
#include "stir/utilities.h"

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <boost/format.hpp>
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::ifstream;
using std::plus;
using std::bind2nd;
#endif

START_NAMESPACE_STIR


/*!
  \brief Tests Array functionality
  \ingroup test
  \warning Running this will create and delete 2 files with names 
  output.flt and output.other. Existing files with these names will be overwritten.

*/
class ArrayTests : public RunTests
{
private:
  // this function tests the next() function and compare it to using full_iterators
  // sadly needs to be declared in the class for VC 6.0
  template <int num_dimensions, class elemT>
  void
  run_tests_on_next(const Array<num_dimensions, elemT>& test)
  {
    // exit if empty array (as do..while() loop would fail)
    if (test.size() == 0)
      return;

    BasicCoordinate<num_dimensions,elemT> index = get_min_indices(test);
    typename Array<num_dimensions, elemT>::const_full_iterator iter = test.begin_all();
    do
      {
        check(*iter == test[index], "test on next(): element out of sequence?");
        ++iter;
      }
    while (next(index, test) && (iter != test.end_all()));
    check (iter == test.end_all(), "test on next() : did we cover all elements?");
  }

  // functions that runs IO tests for an array of arbitrary dimension
  // sadly needs to be declared in the class for VC 6.0
  template <int num_dimensions, typename elemT>
  void run_IO_tests(const Array<num_dimensions, elemT>&t1)
  {
    std::fstream os;
    std::fstream is;
    run_IO_tests_with_file_args(os, is, t1);  
    FILE* ofptr;
    FILE* ifptr;
    run_IO_tests_with_file_args(ofptr, is, t1);  
    run_IO_tests_with_file_args(ofptr, ifptr, t1);  
  }
  template <int num_dimensions, typename elemT,class OFSTREAM, class IFSTREAM>
  void run_IO_tests_with_file_args(OFSTREAM& os, IFSTREAM& is, const Array<num_dimensions, elemT>&t1)
  {
    {
      open_write_binary(os, "output.flt");
      check(write_data(os,t1)==Succeeded::yes, "write_data could not write array");
      close_file(os);
    }
    Array<num_dimensions,elemT> t2(t1.get_index_range());
    {
      open_read_binary(is, "output.flt");
      check(read_data(is,t2)==Succeeded::yes, "read_data could not read from output.flt");
      close_file(is);
    }
    check_if_equal(t1  ,t2, "test out/in" );
    remove("output.flt");

    {
      open_write_binary(os, "output.flt");
      const Array<num_dimensions, elemT> copy=t1;
      check(write_data(os,t1,ByteOrder::swapped)==Succeeded::yes, "write_data could not write array with swapped byte order");
      check_if_equal(t1  ,copy, "test out with byte-swapping didn't change the array" );
      close_file(os);
    }
    {
      open_read_binary(is, "output.flt");
      check(read_data(is,t2,ByteOrder::swapped)==Succeeded::yes, "read_data could not read from output.flt");
      close_file(is);
    }
    check_if_equal(t1  ,t2, "test out/in (swapped byte order)" );
    remove("output.flt");

    cerr <<"\tTests writing as shorts\n";
    run_IO_tests_mixed(os, is, t1, NumericInfo<short>());
    cerr <<"\tTests writing as floats\n";
    run_IO_tests_mixed(os, is, t1, NumericInfo<float>());
    cerr <<"\tTests writing as signed chars\n";
    run_IO_tests_mixed(os, is, t1, NumericInfo<signed char>());


    /* check on failed IO.
       Note: needs to be after the others, as we would have to call os.clear()
       for ostream to be able to write again, but that's not defined for FILE*.
    */
    {
      const Array<num_dimensions, elemT> copy=t1;
      cerr << "\n\tYou should now see a warning that writing failed. That's by intention.\n";
      check(write_data(os,t1,ByteOrder::swapped)!=Succeeded::yes, "write_data with swapped byte order should have failed");
      check_if_equal(t1  ,copy, "test out with byte-swapping didn't change the array even with failed IO" );
    }

  }

  //! function that runs IO tests with mixed types for array of arbitrary dimension
  // sadly needs to be implemented in the class for VC 6.0
  template <int num_dimensions, typename elemT, class OFSTREAM, class IFSTREAM, class output_type>
  void run_IO_tests_mixed(OFSTREAM& os, IFSTREAM& is, const Array<num_dimensions, elemT>&orig, NumericInfo<output_type> output_type_info)
    {
      {
        open_write_binary(os, "output.orig");
        elemT scale(1);
        check(write_data(os, orig, NumericInfo<elemT>(), scale)==Succeeded::yes, "write_data could not write array in original data type");
        close_file(os);
        check_if_equal(scale ,static_cast<elemT>(1), "test out/in: data written in original data type: scale factor should be 1" );
      }
      elemT scale(1);
      bool write_data_ok;
      {
        ofstream os;
        open_write_binary(os, "output.other");
        write_data_ok=check(write_data(os,orig, output_type_info, scale)==Succeeded::yes, "write_data could not write array as other_type");
        close_file(os);
      }

      if (write_data_ok)
        { 
          // only do reading test if data was written
          Array<num_dimensions,output_type> data_read_back(orig.get_index_range());
          {
            open_read_binary(is, "output.other");
            check(read_data(is, data_read_back)==Succeeded::yes, "read_data could not read from output.other");
            close_file(is);        
            remove("output.other");
          }

          // compare with convert()
          {
            float newscale = scale;
            Array<num_dimensions,output_type> origconverted = 
              convert_array(newscale, orig, NumericInfo<output_type>());
            check_if_equal(newscale ,scale, "test read_data <-> convert : scale factor ");
            check_if_equal(origconverted ,data_read_back, "test read_data <-> convert : data");
          }

          // compare orig/scale with data_read_back
          {
            const Array<num_dimensions,elemT> orig_scaled(orig/scale);
            this->check_array_equality_with_rounding(orig_scaled, data_read_back, 
                                                     "test out/in: data written as other_type, read as other_type");
          }

          // compare data written as original, but read as other_type
          {
            Array<num_dimensions,output_type> data_read_back2(orig.get_index_range());
        
            ifstream is;
            open_read_binary(is, "output.orig");
        
            elemT in_scale = 0;
            check(read_data(is, data_read_back2, NumericInfo<elemT>(), in_scale)==Succeeded::yes, "read_data could not read from output.orig");
            // compare orig/in_scale with data_read_back2
            const Array<num_dimensions,elemT> orig_scaled(orig/in_scale);
            this->check_array_equality_with_rounding(orig_scaled, data_read_back2, 
                                                     "test out/in: data written as original_type, read as other_type");
          }
        } // end of if(write_data_ok)
      remove("output.orig");
    }

  //! a special version of check_if_equal just for this class
  /*! we check up to .5 if output_type is integer, and up to tolerance otherwise
   */
  template <int num_dimensions, typename elemT, class output_type>
  bool check_array_equality_with_rounding(const Array<num_dimensions,elemT>& orig, const Array<num_dimensions,output_type>& data_read_back, const char*const message)
  {
    NumericInfo<output_type> output_type_info;
    bool test_failed=false;
    typename Array<num_dimensions,elemT>::const_full_iterator diff_iter = orig.begin_all();
    typename Array<num_dimensions,output_type>::const_full_iterator data_read_back_iter = data_read_back.begin_all_const();
    while(diff_iter!=orig.end_all())
      {
        if (output_type_info.integer_type())
          {
            std::stringstream full_message;
            // construct useful error message even though we use a boolean check
            full_message << boost::format("unequal values are %2% and %3%. %1%: difference larger than .5")
              % message % static_cast<elemT>(*data_read_back_iter) % *diff_iter;
            // difference should be maximum .5 (but we test with slightly larger tolerance to accomodate numerical precision)
            test_failed = check(fabs(*diff_iter - *data_read_back_iter)<=.502, 
                                full_message.str().c_str());
          }
        else
          {
            string full_message = message;
            full_message += ": difference larger than tolerance";
            test_failed = check_if_equal(static_cast<elemT>(*data_read_back_iter), *diff_iter, 
                                         full_message.c_str());
          }
        if (test_failed)
          break;
        diff_iter++; data_read_back_iter++;
      }
    return test_failed;
  }

public:
  void run_tests();
};


void
ArrayTests::run_tests()
{

  cerr << "Testing Array classes\n";
  {
    cerr << "Testing 1D stuff" << endl;

    {
        
      Array<1,int> testint(IndexRange<1>(5));
      testint[0] = 2;
      check_if_equal(testint.size(), size_t(5), "test size()");
      check_if_equal(testint.size_all(), size_t(5), "test size_all()");

      Array<1,float> test(IndexRange<1>(10));
      check_if_zero(test, "Array1D not initialised to 0");

      test[1] = (float)10.5;
      test.set_offset(-1);
      check_if_equal(test.size(), size_t(10), "test size() with non-zero offset");
      check_if_equal(test.size_all(), size_t(10), "test size_all() with non-zero offset");
      check_if_equal( test[0], 10.5F, "test indexing of Array1D");
      test += 1;
      check_if_equal( test[0] , 11.5F, "test operator+=(float)");
      check_if_equal( test.sum(), 20.5F,  "test operator+=(float) and sum()");
      check_if_zero( test - test, "test operator-(Array1D)");

      BasicCoordinate<1,int> c;
      c[1]=0;       
      check_if_equal(test[c] , 11.5F , "test operator[](BasicCoordinate)");   
      test[c] = 12.5;
      check_if_equal(test[c] , 12.5F , "test operator[](BasicCoordinate)");  

      {
        Array<1,float> ref(-1,2); 
        ref[-1]=1.F;ref[0]=3.F;ref[1]=3.14F;
        Array<1,float> test = ref;

        test += 1;
        for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
          check_if_equal( test[i] , ref[i]+1, "test operator+=(float)");
        test = ref; test -= 4;
        for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
          check_if_equal( test[i] , ref[i]-4, "test operator-=(float)");
        test = ref; test *= 3;
        for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
          check_if_equal( test[i] , ref[i]*3, "test operator*=(float)");
        test = ref; test /= 3;
        for (int i=ref.get_min_index(); i<= ref.get_max_index(); ++i)
          check_if_equal( test[i] , ref[i]/3, "test operator/=(float)");
      }
      {
        Array<1,float> test2;
        test2 = test * 2;
        check_if_equal( 2*test[0] , test2[0], "test operator*(float)");
      }

      {
        Array<1,float> test2 = test;
        test.grow(-2,test.get_max_index());
        Array<1,float> test3 = test2 + test;
        check_if_zero(test3[-2], "test growing during operator+");
      }

    }
#if 1    
    {
      // tests on log/exp
      Array<1,float> test(-3,10);
      test.fill(1.F);
      in_place_log(test);
      {
        Array<1,float> testeq(-3,10);
        check_if_equal(test  , testeq, "test in_place_log of Array1D");
      }
      {
        for (int i=test.get_min_index(); i<= test.get_max_index(); i++)
          test[i] = 3.5F*i + 100;
      }
      Array<1,float> test_copy = test;

      in_place_log(test);
      in_place_exp(test);
      check_if_equal(test , test_copy, "test log/exp of Array1D"); 
    }
#endif
  }
   
  {
    cerr << "Testing 2D stuff" << endl;
    {
      const IndexRange<2> range(Coordinate2D<int>(0,0),Coordinate2D<int>(9,9));
      Array<2,float> test2(range);
      check_if_equal(test2.size(), size_t(10), "test size()");
      check_if_equal(test2.size_all(), size_t(100), "test size_all()");
      // KT 17/03/98 added check on initialisation
      check_if_zero(test2, "test Array<2,float> not initialised to 0" );

#if 0
      // KT 06/04/98 removed operator()
      test2(3,4) = (float)23.3;
#else
      test2[3][4] = (float)23.3;
#endif
      //test2.set_offsets(-1,-4);
      //check_if_equal( test2[2][0] , 23.3, "test indexing of Array2D");
    }


    {
      IndexRange<2>  range(Coordinate2D<int>(0,0),Coordinate2D<int>(3,3));
      Array<2,float> testfp(range); 
      Array<2,float> t2fp(range);
#if 0
      // KT 06/04/98 removed operator()
      testfp(3,2) = 3.3F;
      t2fp(3,2) = 2.2F;
#else
      testfp[3][2] = 3.3F;
      t2fp[3][2] = 2.2F;
#endif

      Array<2,float> t2 = t2fp + testfp;
      check_if_equal( t2[3][2] , 5.5F, "test operator +(Array2D)");
      t2fp += testfp;
      check_if_equal( t2fp[3][2] , 5.5F, "test operator +=(Array2D)");
      check_if_equal(t2  , t2fp, "test comparing Array2D+= and +" );

      {     
        BasicCoordinate<2,int> c;
        c[1]=3; c[2]=2; 
        check_if_equal(t2[c], 5.5F, "test on operator[](BasicCoordinate)");   
        t2[c] = 6.;
        check_if_equal(t2[c], 6.F, "test on operator[](BasicCoordinate)");   
      }

      // assert should break on next line (in Debug build) if uncommented
      //t2[-4][3]=1.F;
      // at() should throw error
      {
        bool exception_thrown=false;
        try
          {
            t2.at(-4).at(3);
          }
        catch (...)
          {
            exception_thrown=true;
          }
        check(exception_thrown, "out-of-range index should throw an exception");
      }
      
      //t2.grow_height(-5,5);
      IndexRange<2> larger_range(Coordinate2D<int>(-5,0),Coordinate2D<int>(5,3));
      t2.grow(larger_range);
      t2[-4][3]=1.F;
      check_if_equal( t2[3][2] , 6.F, "test on grow");
    
      // test assignment
      t2fp = t2;
      check_if_equal(t2  , t2fp, "test operator=(Array2D)" );

      {
        Array<2,float> tmp;
        tmp = t2 / 2;
        check_if_equal( t2.sum()/2 , tmp.sum(), "test operator/(float)");
      }

      {
        // copy constructor;
        Array<2,float> t21(t2);
        check_if_equal(t21  , t2, "test Array2D copy constructor" );
        // 'assignment constructor' (this simply calls copy constructor)
        Array<2,float> t22 = t2;
        check_if_equal(t22  , t2, "test Array2D copy constructor" );
      }
    }
    // size_all with irregular range
    {
      const IndexRange<2> range(Coordinate2D<int>(-1,1),Coordinate2D<int>(1,2));
      Array<2,float> test2(range);
      check(test2.is_regular(), "test is_regular() with regular");
      check_if_equal(test2.size(), size_t(3), "test size() with non-zero offset");
      check_if_equal(test2.size_all(), size_t(6), "test size_all() with non-zero offset");  
      test2[0].resize(-1,2);
      check(!test2.is_regular(), "test is_regular() with irregular");
      check_if_equal(test2.size(), size_t(3), "test size() with irregular range");
      check_if_equal(test2.size_all(), size_t(6+2), "test size_all() with irregular range");
    }
    // full iterator
    {
      IndexRange<2> range(Coordinate2D<int>(0,0),Coordinate2D<int>(2,2));
      Array<2,float> test2(range);
      {
        float value = 1.2F;
        for (Array<2,float>::full_iterator iter = test2.begin_all();
             iter != test2.end_all(); 
             )
          *iter++ = value++;
      }
      {
        float value = 1.2F;
        Array<2,float>::const_full_iterator iter = test2.begin_all_const();
        for (int i=test2.get_min_index(); i<= test2.get_max_index(); ++i)
          for (int j=test2[i].get_min_index(); j<= test2[i].get_max_index(); ++j)
            {
              check(iter != test2.end_all_const(), "test on 2D full iterator");
              check_if_equal(*iter++, test2[i][j], "test on 2D full iterator vs. index");
              check_if_equal(test2[i][j], value++, "test on 2D full iterator value");
            }
      }

      const Array<2,float> empty;
      check(empty.begin_all() == empty.end_all(), "test on 2D full iterator for empty range");
    }
    // tests for next()
    {
      const IndexRange<2> range(Coordinate2D<int>(-1,1),Coordinate2D<int>(1,2));
      Array<2,int> test(range);
      // fill array with numbers in sequence
      {
        Array<2,int>::full_iterator iter = test.begin_all();
        for (int i=0; iter!= test.end_all(); ++iter, ++i)
          {
            *iter = i;
          }
      }
      std::cerr << "\tTest on next() with regular array\n";
      this->run_tests_on_next(test);
      // now do test with irregular array
      test[0].resize(0,2);
      test[0][2] = 10;
      std::cerr << "\tTest on next() with irregular array, case 1\n";
      this->run_tests_on_next(test);
      test[1].resize(-2,2);
      test[1][-2] = 20;
      std::cerr << "\tTest on next() with irregular array, case 2\n";
      this->run_tests_on_next(test);
      test[-1].resize(-2,0);
      test[-1][-2] = 30;
      std::cerr << "\tTest on next() with irregular array, case 3\n";
      this->run_tests_on_next(test);
    }
  }

  {
    cerr << "Testing 3D stuff" << endl;

    IndexRange<3> range(Coordinate3D<int>(0,-1,1),Coordinate3D<int>(3,3,3));
    Array<3,float> test3(range);
    check_if_equal(test3.size(), size_t(4), "test size()");
    check_if_equal(test3.size_all(), size_t(60), "test size_all() with non-zero offset");
    // KT 06/04/98 removed operator()
#if 0
    test3(1,2,1) = (float)6.6;
#else
    test3[1][2][1] = (float)6.6;
#endif
    test3[1][0][2] = (float)7.3;
    test3[1][0][1] = -1;

    
    check_if_equal( test3.sum() , 12.9F, "test on sum");
    check_if_equal( test3.find_max() , 7.3F, "test on find_max");
    check_if_equal( test3.find_min() , -1.F, "test on find_min");

    {
       Array<3,float> test3copy(test3);
       BasicCoordinate<3,int> c;
       c[1]=1; c[2]=0; c[3]=2;
       check_if_equal(test3[c], 7.3F, "test on operator[](BasicCoordinate)");   
       test3copy[c]=8.;
       check_if_equal(test3copy[1][0][2], 8.F, "test on operator[](BasicCoordinate)");   
    }

    Array<3,float> test3bis(range);
    test3bis[1][2][1] = (float)6.6;
    test3bis[1][0][1] = (float)1.3;
    Array<3,float> test3ter = test3bis;

    test3ter += test3;
    check_if_equal(test3ter[1][0][1] , .3F, "test on operator+=(Array3D)");

    Array<3,float> test3quat = test3 + test3bis;
    check_if_equal(test3quat  , test3ter, "test summing Array3D");

    {
      Array<3,float> tmp= test3 - 2;
      Array<3,float> tmp2 = test3;
      tmp2.fill(1.F);
      
      check_if_zero( test3.sum() - 2*tmp2.sum() - tmp.sum(), "test operator-(float)");
    }

#if !defined(_MSC_VER) || _MSC_VER>1300
    // VC 6.0 cannot compile this
    in_place_apply_function(test3ter, bind2nd(plus<float>(), 4.F));
    test3quat += 4.F;
    check_if_equal(test3quat  , test3ter, 
                  "test in_place_apply_function and operator+=(NUMBER)");
#endif
    // size_all with irregular range
    {
      const IndexRange<3> range(Coordinate3D<int>(-1,1,4),Coordinate3D<int>(1,2,6));
      Array<3,float> test(range);
      check(test.is_regular(), "test is_regular() with regular");
      check_if_equal(test.size(), size_t(3), "test size() with non-zero offset");
      check_if_equal(test.size_all(), size_t(3*2*3), "test size_all() with non-zero offset");  
      test[0][1].resize(-1,2);
      check(!test.is_regular(), "test is_regular() with irregular");
      check_if_equal(test.size(), size_t(3), "test size() with irregular range");
      check_if_equal(test.size_all(), size_t(3*2*3+4-3), "test size_all() with irregular range");
    }
    // full iterator
    {
      IndexRange<3> range(Coordinate3D<int>(0,0,1),Coordinate3D<int>(2,2,3));
      Array<3,float> test(range);
      {
        float value = 1.2F;
        for (Array<3,float>::full_iterator iter = test.begin_all();
             iter != test.end_all(); 
             )
          *iter++ = value++;
      }
      {
        float value = 1.2F;
        Array<3,float>::const_full_iterator iter = test.begin_all_const();
        for (int i=test.get_min_index(); i<= test.get_max_index(); ++i)
          for (int j=test[i].get_min_index(); j<= test[i].get_max_index(); ++j)
            for (int k=test[i][j].get_min_index(); k<= test[i][j].get_max_index(); ++k)
            {
              check(iter != test.end_all_const(), "test on 3D full iterator");
              check_if_equal(*iter++, test[i][j][k], "test on 3D full iterator vs. index");
              check_if_equal(test[i][j][k], value++, "test on 3D full iterator value");
            }
      }
      // test empty container
      {
        const Array<3,float> empty;
        check(empty.begin_all() == empty.end_all(), "test on 3D full iterator for empty range");
      }
      // test conversion from full_iterator to const_full_iterator
      {
        Array<3,float>::full_iterator titer= test.begin_all();
        Array<3,float>::const_full_iterator ctiter= titer; // this should compile
      }
    }
  }


  {
    cerr << "Testing 4D stuff" << endl;
    const IndexRange<4> range(Coordinate4D<int>(-3,0,-1,1),Coordinate4D<int>(-2,3,3,3));
    Array<4,float> test4(range);
    test4.fill(1.);
    test4[-3][1][2][1] = (float)6.6;
#if 0
    test4(-2,1,0,2) = (float)7.3;
#else
    test4[-2][1][0][2] = (float)7.3;
#endif
    {
    float sum = test4.sum();
    check_if_equal( sum , 131.9F, "test on sum()");
    }
    const IndexRange<4> larger_range(Coordinate4D<int>(-3,0,-1,1),Coordinate4D<int>(-1,3,3,5));
    test4.grow(larger_range);
    check_if_equal(test4.get_index_range(), larger_range, "test Array4D grow index range");
    check_if_equal(test4.sum(), 131.9F , "test Array4D grow sum");
    {
      const Array<4,float> test41 = test4;
      check_if_equal(test4  , test41, "test Array4D copy constructor" );
      check_if_equal( test41[-3][1][2][1] , 6.6F, "test on indexing after grow");
    }
    {
      Array<4,float> test41 = test4;
      const IndexRange<4> mixed_range(Coordinate4D<int>(-4,1,0,1),Coordinate4D<int>(-2,3,3,6));
      test41.resize(mixed_range);
      check_if_equal(test41.get_index_range(), mixed_range, "test Array4D resize index range");
      check_if_equal( test41[-3][1][2][1] , 6.6F, "test on indexing after resize");
    }
    { 
      BasicCoordinate<4,int> c;
      c[1]=-2;c[2]=1;c[3]=0;c[4]=2;
      check_if_equal(test4[c] , 7.3F , "test on operator[](BasicCoordinate)");   
      test4[c]=1.;
      check_if_equal(test4[c] , 1.F , "test on operator[](BasicCoordinate)");   
    }
    {
      Array<4,float> test4bis(range);
      test4bis[-2][1][2][1] = (float)6.6;
      test4bis[-3][1][0][1] = (float)1.3;
      Array<4,float> test4ter = test4bis;

      test4ter += test4;
      check_if_equal(test4ter[-3][1][0][1] ,2.3F, "test on operator+=(Array4D)");
      check(test4ter.get_index_range() == larger_range, "test range for operator+=(Array4D) with grow");
           
      // Note that test4 is bigger in size than test4bis.
      Array<4,float> test4quat = test4bis + test4;
      check_if_equal(test4quat  ,test4ter, "test summing Array4D with grow");
      check(test4quat.get_index_range() == larger_range, "test range for operator+=(Array4D)");
    }

    // test on scalar multiplication, division
    {
      Array<4,float> test4bis = test4;
      test4bis *= 6.F;
      check_if_equal(test4bis.sum() ,test4.sum()*6, "test operator *=(float)");
      test4bis /= 5.F;
      check_if_equal(test4bis.sum() ,test4.sum()*6.F/5, "test operator /=(float)");
    } 

    // test on element-wise multiplication, division
    {
      Array<4,float> test4bis(range);
      {
        for (int i=test4bis.get_min_index(); i<= test4bis.get_max_index(); i++)
          test4bis[i].fill(i+10.F);
      }
      // save for comparison later on
      Array<4,float> test4ter = test4bis;
      
      // Note that test4 is bigger than test4bis, so it will grow with the *=
      // new elements in test4bis will remain 0 because we're using multiplication
      test4[-1].fill(666);
      test4bis *= test4;
      check_if_zero(test4bis[-1], "test operator *=(Array4D) grows ok");

      check(test4.get_index_range() == test4bis.get_index_range(), "test operator *=(Array4D) grows ok: range");
      // compute the new sum. 
      {
        float sum_check = 0;      
        for (int i=test4.get_min_index(); i<= -2; i++)
          sum_check += test4[i].sum()*(i+10.F);
        check_if_equal(test4bis.sum() ,sum_check, "test operator *=(Array4D)");
      }
      // divide test4, but add a tiny number to avoid division by zero
      const Array<4,float> test4quat = test4bis / (test4+.00001F);
      test4ter.grow(test4.get_index_range());
      check_if_equal(test4ter ,test4quat, "test operator /(Array4D)");
    } 
  
    // test operator+(float)
    {
      // KT 31/01/2000 new
      Array<4,float> tmp= test4 + 2;
      Array<4,float> tmp2 = test4;
      tmp2.fill(1.F);
      
      // KT 20/12/2001 made check_if_zero compare relative to 1 by dividing
      check_if_zero( (test4.sum() + 2*tmp2.sum() - tmp.sum())/test4.sum(), 
                     "test operator+(float)");
    }
  }

#if 1
  {
    cerr << "Testing 1D float IO" << endl;
    Array<1,float> t1(IndexRange<1>(-1,10));
    for (int i=-1; i<=10; i++)
      t1[i] = static_cast<float>(sin(i* _PI/ 15.));
        run_IO_tests(t1);
  }
  {
    cerr << "Testing 2D double IO" << endl;
    IndexRange<2> range(Coordinate2D<int>(-1,11),Coordinate2D<int>(10,20));
    Array<2,double> t1(range);
    for (int i=-1; i<=10; i++)
      for (int j=11; j<=20; j++)
        t1[i][j] = static_cast<double>(sin(i*j* _PI/ 15.));
    run_IO_tests(t1);
  }
  {
    cerr << "Testing 3D float IO" << endl;

    // construct test array which has rows of very different magnitudes,
    // numbers in last rows do not fit into short integers
    IndexRange<3> range(Coordinate3D<int>(-1,11,21),Coordinate3D<int>(10,20,30));
    Array<3,float> t1(range);
    for (int i=-1; i<=10; i++)
      for (int j=11; j<=20; j++)
        for (int k=21; k<=30; k++)
          t1[i][j][k] = static_cast<float>(20000.*k*sin(i*j*k* _PI/ 3000.));
    run_IO_tests(t1);
  }
#endif
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  ArrayTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
