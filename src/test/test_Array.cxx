// $Id$

/*!
  \file 
  \ingroup test
 
  \brief tests for the Array class

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

// set to high level of debugging
#ifdef _DEBUG
#undef _DEBUG
#endif
#define _DEBUG 2

#include "stir/Array.h"
#include "stir/Coordinate2D.h"
#include "stir/Coordinate3D.h"
#include "stir/Coordinate4D.h"
#include "stir/convert_array.h"

#include "stir/RunTests.h"

#include "stir/ArrayFunction.h"
#include <functional>

// for open_read/write_binary
#include "stir/utilities.h"

// for 'remove'
#include <cstdio>
#include <fstream>

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
  output.flt and output.short. Existing files with these names will be overwritten.

*/
class ArrayTests : public RunTests
{
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

      Array<1,float> test(IndexRange<1>(10));
      check_if_zero(test, "Array1D not initialised to 0");

      test[1] = (float)10.5;
      test.set_offset(-1);
      check_if_equal( test[0], 10.5, "test indexing of Array1D");
      test += 1;
      check_if_equal( test[0] , 11.5, "test operator+=(float)");
      check_if_equal( test.sum(), 20.5,  "test operator+=(float) and sum()");
    
      check_if_zero( test - test, "test operator-(Array1D)");

      {
	// KT 31/01/2000 new
	Array<1,float> test2;
	test2 = test * 2;
	check_if_equal( 2*test[0] , test2[0], "test operator*(float)");
      }

      {
	// KT 31/01/2000 new
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
	  test[i] = 3.5*i + 100;
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
      IndexRange<2> range(Coordinate2D<int>(0,0),Coordinate2D<int>(9,9));
      Array<2,float> test2(range);
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
      check_if_equal( t2[3][2] , 5.5, "test operator +(Array2D)");
      t2fp += testfp;
      check_if_equal( t2fp[3][2] , 5.5, "test operator +=(Array2D)");
      check_if_equal(t2  , t2fp, "test comparing Array2D+= and +" );

      // assert should break on next line if uncommented
      //t2[-4][3]=1.F;

      //t2.grow_height(-5,5);
      IndexRange<2> larger_range(Coordinate2D<int>(-5,0),Coordinate2D<int>(5,3));
      t2.grow(larger_range);
      t2[-4][3]=1.F;
      check_if_equal( t2[3][2] , 5.5);
    
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

    // full iterator
#ifdef FULL
    {
      IndexRange<2> range(Coordinate2D<int>(0,0),Coordinate2D<int>(2,2));
      Array<2,float> test2(range);
      test2.fill(1);

      
      for (Array<2,float>::full_iterator iter = test2.begin_all();
           iter != test2.end_all(); 
	   ++iter)
	cerr << *iter <<" ";
      cerr << endl;

      float value = 1.2F;
      for (Array<2,float>::full_iterator iter = test2.begin_all();
           iter != test2.end_all(); 
	   )
	 *iter++ = value++;
	   
      for (Array<2,float>::full_iterator iter = test2.begin_all();
           iter != test2.end_all(); 
	   ++iter)
	cerr << *iter <<" ";
      cerr << endl;


      Array<2,float> empty;
      if (empty.begin_all() != empty.end_all())
	cerr << "Not equal"<<endl;

    }
#endif
  }

  {
    cerr << "Testing 3D stuff" << endl;

    IndexRange<3> range(Coordinate3D<int>(0,-1,1),Coordinate3D<int>(3,3,3));
    Array<3,float> test3(range);
    // KT 06/04/98 removed operator()
#if 0
    test3(1,2,1) = (float)6.6;
#else
    test3[1][2][1] = (float)6.6;
#endif
    test3[1][0][2] = (float)7.3;
    test3[1][0][1] = -1;
    
    check_if_equal( test3.sum() , 12.9, "test on sum");
    check_if_equal( test3.find_max() , 7.3, "test on find_max");
    check_if_equal( test3.find_min() , -1., "test on find_min");

    Array<3,float> test3bis(range);
    test3bis[1][2][1] = (float)6.6;
    test3bis[1][0][1] = (float)1.3;
    Array<3,float> test3ter = test3bis;

    test3ter += test3;
    check_if_equal(test3ter[1][0][1] , .3, "test on operator+=(Array3D)");

    Array<3,float> test3quat = test3 + test3bis;
    check_if_equal(test3quat  , test3ter, "test summing Array3D");

    {
      Array<3,float> tmp= test3 - 2;
      Array<3,float> tmp2 = test3;
      tmp2.fill(1.F);
      
      check_if_zero( test3.sum() - 2*tmp2.sum() - tmp.sum(), "test operator-(float)");
    }

#if !defined(_MSC_VER)
    // VC cannot compile this
    in_place_apply_function(test3ter, bind2nd(plus<float>(), 4.F));
    test3quat += 4.F;
    check_if_equal(test3quat  , test3ter, 
		  "test in_place_apply_function and operator+=(NUMBER)");
#endif

    // full iterator
#ifdef FULL
    {
      IndexRange<3> range(Coordinate3D<int>(0,0,0),Coordinate3D<int>(1,1,1));
      Array<3,float> test3(range);
      test3.fill(1);

      
      for (Array<3,float>::full_iterator iter = test3.begin_all();
           iter != test3.end_all(); 
	   ++iter)
	cerr << *iter <<" ";
      cerr << endl;
    }
#endif    
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
    check_if_equal( sum , 131.9, "test on sum()");
    }
    const IndexRange<4> larger_range(Coordinate4D<int>(-3,0,-1,1),Coordinate4D<int>(-1,3,3,3));
    test4.grow(larger_range);
    Array<4,float> test41 = test4;
    check_if_equal(test4  , test41, "test Array4D copy constructor" );
    check_if_equal( test41[-3][1][2][1] , 6.6, "test on indexing after grow");

    {
      Array<4,float> test4bis(range);
      test4bis[-2][1][2][1] = (float)6.6;
      test4bis[-3][1][0][1] = (float)1.3;
      Array<4,float> test4ter = test4bis;

      test4ter += test4;
      check_if_equal(test4ter[-3][1][0][1] ,2.3, "test on operator+=(Array4D)");

      // Note that test4 is bigger in size than test4bis.
      Array<4,float> test4quat = test4bis + test4;
      check_if_equal(test4quat  ,test4ter, "test summing Array4D with grow");
    }

    // test on scalar multiplication, division
    {
      Array<4,float> test4bis = test4;
      test4bis *= 6.F;
      check_if_equal(test4bis.sum() ,test4.sum()*6, "test operator *=(float)");
      test4bis /= 5.F;
      check_if_equal(test4bis.sum() ,test4.sum()*6./5, "test operator /=(float)");
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
      const Array<4,float> test4quat = test4bis / test4;
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

  {
    cerr << "Testing 1D IO" << endl;
    Array<1,float> t1(IndexRange<1>(-1,10));
    for (int i=-1; i<=10; i++)
      t1[i] = sin(i* _PI/ 15.);

    ofstream os;
    open_write_binary(os, "output.flt");
    t1.write_data(os);
    os.close();

    Array<1,float> t2(IndexRange<1>(-1,10));
    ifstream is;
    open_read_binary(is, "output.flt");
    t2.read_data(is);

    check_if_equal(t1  ,t2 , "test 1D out/in" );    

    // KT 31/01/2000 new
    // byte-swapped 1D IO    
    {
      ofstream os;
      open_write_binary(os, "output.flt");
      t1.write_data(os, ByteOrder::swapped);
      os.close();
      
      Array<1,float> t2(IndexRange<1>(-1,10));
      ifstream is;
      open_read_binary(is, "output.flt");
      t2.read_data(is, ByteOrder::swapped);
      
      check_if_equal(t1  ,t2 , "test 1D out/in with byte-swapping" );    
    }
  }
 
  
  {
    cerr << "Testing 2D IO" << endl;
    IndexRange<2> range(Coordinate2D<int>(-1,11),Coordinate2D<int>(10,20));
    Array<2,float> t1(range);
    for (int i=-1; i<=10; i++)
      for (int j=11; j<=20; j++)
	t1[i][j] = sin(i*j* _PI/ 15.);

    ofstream os;
    open_write_binary(os, "output.flt");
    t1.write_data(os);
    os.close();

    Array<2,float> t2(range);
    ifstream is;
    open_read_binary(is, "output.flt"); 
    t2.read_data(is);

    check_if_equal(t1  ,t2, "test 2D out/in" );
  }
  
  {
    cerr << "Testing 3D IO" << endl;
    IndexRange<3> range(Coordinate3D<int>(-1,11,21),Coordinate3D<int>(10,20,30));
    Array<3,float> t1(range);
    for (int i=-1; i<=10; i++)
      for (int j=11; j<=20; j++)
	for (int k=21; k<=30; k++)
	  t1[i][j][k] = static_cast<float>(sin(i*j*k* _PI/ 15.));

    ofstream os;
    open_write_binary(os, "output.flt");
    t1.write_data(os);
    os.close();

    Array<3,float> t2(range);
    ifstream is;
    open_read_binary(is, "output.flt");
    t2.read_data(is);

    check_if_equal(t1  ,t2, "test 3D out/in" );
  }

#if !defined(_MSC_VER) || (_MSC_VER > 1100)
  {
    cerr << "Testing 3D IO in different data types" << endl;

    // construct test tensor which has rows of very different magnitudes,
    // numbers in last rows do not fit into short integers
    IndexRange<3> range(Coordinate3D<int>(-1,11,21),Coordinate3D<int>(10,20,30));
    Array<3,float> floats(range);
    for (int i=-1; i<=10; i++)
      for (int j=11; j<=20; j++)
	for (int k=21; k<=30; k++)
	  floats[i][j][k] = static_cast<float>(20000.*k*sin(i*j*k* _PI/ 3000.));

    {
      ofstream os;
      open_write_binary(os, "output.flt");
      float scale = 1;
      floats.write_data(os, NumericType::FLOAT, scale);
      check_if_equal(scale ,1., "test 3D out/in: floats written as floats" );
    }


    float scale = 1;
    {
      ofstream os;
      open_write_binary(os, "output.short");
      floats.write_data(os, NumericType::SHORT, scale);
    }
    assert(scale>1);
    

    Array<3,short> shorts(floats.get_index_range());
    {
      ifstream is;
      open_read_binary(is, "output.short");
      shorts.read_data(is);
    }

    // compare write_data of floats as shorts with convert()
    {
      float newscale = scale;
      Array<3,short> floatsconverted = convert_array(newscale, floats, NumericInfo<short>());
      check_if_equal(newscale ,scale, "test read_data <-> convert : scale factor ");
      check_if_equal(floatsconverted ,shorts, "test read_data <-> convert : data");
    }

    // compare floats with shorts*scale
    {
      Array<3,float> diff = floats;
      diff /= scale;
      for (int i1=floats.get_min_index(); i1<= floats.get_max_index(); i1++)
        for (int i2=floats[i1].get_min_index(); i2<= floats[i1].get_max_index(); i2++)
          for (int i3=floats[i1][i2].get_min_index(); i3<= floats[i1][i2].get_max_index(); i3++)
             diff[i1][i2][i3] -= shorts[i1][i2][i3];
	 
      // difference should be maximum .5
      // the next test relies on how check_if_zero works
      diff *= float(2*get_tolerance());
      check_if_zero(diff, "test 3D out/in: floats written as shorts" );
    }


    // compare read_data of floats as shorts with above
    {
      Array<3,short> shorts2(floats.get_index_range());
    
      ifstream is;
      open_read_binary(is, "output.flt");

      float in_scale = 0;
      shorts2.read_data(is, NumericType::FLOAT, in_scale);
      check_if_equal(scale ,in_scale, "test 3D out/in: floats read as shorts: scale" );
      check_if_equal(shorts ,shorts2, "test 3D out/in: floats read as shorts: data" );
    }


  }
#endif
  // clean up test files
  remove("output.flt");
  remove("output.short");
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  ArrayTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
