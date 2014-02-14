//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
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
  \brief tests for the stir::convert_array functions

  \author Kris Thielemans
  \author PARAPET project

*/

#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericInfo.h"
#include "stir/IndexRange3D.h"
#include "stir/RunTests.h"
#include <vector>
#include <iostream>
#include <math.h>

START_NAMESPACE_STIR

//! tests  convert_array functionality
class convert_array_Tests : public RunTests
{
public:
  void run_tests();
};


void
convert_array_Tests::run_tests()
{

  cerr << "Test program for 'convert_array'." << endl 
    << "Everything is fine when there is no output below." << endl;
  
  // 1D
  {
    Array<1,float> tf1(1,20);
    tf1.fill(100.F);
    
    Array<1,short> ti1(1,20);
    ti1.fill(100);
    
    {
      // float -> short with a preferred scale factor
      float scale_factor = float(1);
      Array<1,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
      
      check(scale_factor == float(1),"test convert_array float->short 1D");
      check_if_equal(ti1, ti2, "test convert_array float->short 1D");
    }
    
    
    {
      // float -> short with automatic scale factor
      float scale_factor = 0;
      Array<1,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
      
      check(fabs(NumericInfo<short>().max_value()/1.01 / ti2[1] -1) < 1E-4);
      for (int i=1; i<= 20; i++)
	ti2[i] = short( double(ti2[i]) *scale_factor);
      check(ti1 == ti2);
    }
    
    tf1 *= 1E20F;
    {
      // float -> short with a preferred scale factor that needs to be adjusted
      float scale_factor = 1;
      Array<1,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
      
      check(fabs(NumericInfo<short>().max_value()/1.01 / ti2[1] -1) < 1E-4);
      for (int i=1; i<= 20; i++)
	check(fabs(double(ti2[i]) *scale_factor / tf1[i] - 1) < 1E-4) ;
      
    }
    
    {
      // short -> float with a scale factor = 1
      float scale_factor = 1;
      Array<1,float> tf2 = convert_array(scale_factor, ti1, NumericInfo<float>());
      Array<1,short> ti2(1,20);
      
      
      check(scale_factor == float(1));
      check(tf2[1] == 100.F);
      for (int i=1; i<= 20; i++)
	ti2[i] = short(double(tf2[i]) *scale_factor) ;
      check(ti1 == ti2);
    }
    
    {
      // short -> float with a preferred scale factor = .01
      float scale_factor = .01F;
      Array<1,float> tf2 = convert_array(scale_factor, ti1, NumericInfo<float>());
      Array<1,short> ti2(1,20);
      
      check(scale_factor == float(.01));
      //TODO double->short
      for (int i=1; i<= 20; i++)
	ti2[i] = short(double(tf2[i]) *scale_factor + 0.5) ;
      check(ti1 == ti2);
    }
    
    tf1.fill(-3.2F);
    ti1.fill(-3);
    {
      // positive float -> unsigned short with a preferred scale factor
      float scale_factor = 1;
      Array<1,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
      
      check(scale_factor == float(1));
      check(ti1 == ti2);
    }
    
    {
      Array<1,unsigned short> ti3(1,20);
      ti3.fill(0);
      
      // negative float -> unsigned short with a preferred scale factor
      float scale_factor = 1;
      Array<1,unsigned short> ti2 =
	convert_array(scale_factor, tf1, NumericInfo<unsigned short>());
      
      check(scale_factor == float(1));
      check(ti3 == ti2);
    }
  }
  //   3D

  {
    Array<3,float> tf1(IndexRange3D(1,30,1,182,-2,182));
    tf1.fill(100.F);
    
    Array<3,short> ti1(tf1.get_index_range());
    ti1.fill(100);
    
    {
      // float -> short with a preferred scale factor
      float scale_factor = float(1);
      Array<3,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
      
      check(scale_factor == float(1));
      check(ti1 == ti2);
    }
    
    {
      // float -> short with automatic scale factor
      float scale_factor = 0;
      Array<3,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());
#ifndef DO_TIMING_ONLY      
      check(fabs(NumericInfo<short>().max_value()/1.01 / (*ti2.begin_all()) -1) < 1E-4);
      const Array<3,short>::full_iterator iter_end= ti2.end_all();
      for (Array<3,short>::full_iterator iter= ti2.begin_all();
           iter != iter_end;
	   ++iter)
	*iter = short( double((*iter)) *scale_factor);
      check(ti1 == ti2);
#endif
    }
    
    tf1 *= 1E20F;
    {
      // float -> short with a preferred scale factor that needs to be adjusted
      float scale_factor = 1;
      Array<3,short> ti2 = convert_array(scale_factor, tf1, NumericInfo<short>());

#ifndef DO_TIMING_ONLY
      check(fabs(NumericInfo<short>().max_value()/1.01 / (*ti2.begin_all()) -1) < 1E-4);
      Array<3,short>::full_iterator iter_ti2= ti2.begin_all();
      const Array<3,short>::full_iterator iter_ti2_end= ti2.end_all();
      Array<3,float>::full_iterator iter_tf1= tf1.begin_all();
      for (;
           iter_ti2 != iter_ti2_end;
	   ++iter_ti2, ++iter_tf1)
	check(fabs(double(*iter_ti2) *scale_factor / *iter_tf1 - 1) < 1E-4) ;
#endif      
    }
  }
  // tests on convert_range
  {
    std::vector<signed char> vin(10,2);
    std::vector<int> vout(10);
    float scale_factor=0;
    convert_range(vout.begin(), scale_factor, vin.begin(), vin.end());
    {
      std::vector<int>::const_iterator iter_out= vout.begin();
      std::vector<signed char>::const_iterator iter_in= vin.begin();
      for (;
	   iter_out != vout.end();
	   ++iter_in, ++iter_out)
	check(fabs(double(*iter_out) *scale_factor / *iter_in - 1) < 1E-4,
	      "convert_range signed char->int") ;
    }
  }
  // equal type
  {
    std::vector<int> vin(10,2);
    std::vector<int> vout(10);
    float scale_factor=3;
    convert_range(vout.begin(), scale_factor, vin.begin(), vin.end());
    {
      check_if_equal(scale_factor, 1.F, "scale_factor should be 1 when using equal types");
      std::vector<int>::const_iterator iter_out= vout.begin();
      std::vector<int>::const_iterator iter_in= vin.begin();
      for (;
	   iter_out != vout.end();
	   ++iter_in, ++iter_out)
	check(fabs(double(*iter_out) *scale_factor / *iter_in - 1) < 1E-4,
	      "convert_range equal types") ;
    }
  }
}

END_NAMESPACE_STIR



USING_NAMESPACE_STIR



int main()
{
  convert_array_Tests tests;
  tests.run_tests();
  return tests.main_return_value();
}
