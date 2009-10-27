//
// $Id$
//
/*!
  \file 
  \ingroup numerics_test
  \brief tests the implementation of the IR_filters

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
/*
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

  
#include "stir/RunTests.h"
#include "stir/numerics/IR_filters.h"
#include <vector>
#include <iostream>


#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup numerics_test
  \brief A simple class to test the IIR_filter and FIR_filter function.
*/
class IR_filterTests : public RunTests
{
public:
  IR_filterTests() 
  {}
  void run_tests();
private:
  //istream& in;
};


void IR_filterTests::run_tests()
{  
  cerr << "Testing IR_filter function..." << endl;

  set_tolerance(0.001);
 
  std::vector<int>  input_test_signal, input_factors, poles, 
    output_FIR_test_value, output_IIR_test_value,
    stir_FIR_output, stir_IIR_output;
          
  for (int i=1, imax=11; i<imax ;++i)
    {
      input_test_signal.push_back(i);
                  
      stir_FIR_output.push_back(0);
      stir_IIR_output.push_back(0);
                  
                 
      input_factors.push_back(imax-i);  
      if (i==10)
        break;
                  
      if (i<=5)
        poles.push_back(1);
      else
        poles.push_back(-1);
                          
      /* For Checking different sizes of the poles and input_factors

      if (i==1)
      {
      poles.push_back(1);
      input_factors.push_back(10);      
      }
      if (i==2)
      {
      poles.push_back(1);
      input_factors.push_back(9);       
      }
      if (i==3)
      input_factors.push_back(8);
      */
    }

  output_FIR_test_value.push_back(10); //1
  output_FIR_test_value.push_back(29); //2
  output_FIR_test_value.push_back(56); //3
  output_FIR_test_value.push_back(90); //4
  output_FIR_test_value.push_back(130); //5
  output_FIR_test_value.push_back(175); //6
  output_FIR_test_value.push_back(224); //7
  output_FIR_test_value.push_back(276); //8
  output_FIR_test_value.push_back(330); //9
  output_FIR_test_value.push_back(385); //10

  output_IIR_test_value.push_back(10); //1
  output_IIR_test_value.push_back(19); //2
  output_IIR_test_value.push_back(27); //3
  output_IIR_test_value.push_back(34); //4
  output_IIR_test_value.push_back(40); //5
  output_IIR_test_value.push_back(45); //6
  output_IIR_test_value.push_back(69); //7
  output_IIR_test_value.push_back(90); //8
  output_IIR_test_value.push_back(108); //9
  output_IIR_test_value.push_back(123); //10


  {   
    cerr << "Testing FIR_filter function..." << endl;   
                  
    FIR_filter(stir_FIR_output.begin(), stir_FIR_output.end(),
               input_test_signal.begin(), input_test_signal.end(),
               input_factors.begin(), input_factors.end(),0);
                  
    std::vector<int>:: iterator cur_iter_stir_out= stir_FIR_output.begin(), 
      cur_iter_FIR_out= output_FIR_test_value.begin();

    for (;
         cur_iter_stir_out!=stir_FIR_output.end() 
           &&  
           cur_iter_FIR_out!=output_FIR_test_value.end();
         ++cur_iter_stir_out,  ++cur_iter_FIR_out)
      check_if_equal(*cur_iter_stir_out, *cur_iter_FIR_out,
                     "check FIR filter implementation");                  
  }     
  {    
    cerr << "Testing IIR_filter function..." << endl;   
    IIR_filter(stir_IIR_output.begin(), stir_IIR_output.end(),
               input_test_signal.begin(), input_test_signal.end(),
               input_factors.begin(), input_factors.end(),
               poles.begin(), poles.end(),0);             
    std::vector<int>:: iterator cur_iter_stir_IIR_out= 
      stir_IIR_output.begin(), 
      cur_iter_IIR_out= output_IIR_test_value.begin();
    for (;
         cur_iter_stir_IIR_out!=stir_IIR_output.end() 
           &&  
           cur_iter_IIR_out!=output_IIR_test_value.end();
         ++cur_iter_stir_IIR_out,  ++cur_iter_IIR_out)
      check_if_equal(*cur_iter_stir_IIR_out, *cur_iter_IIR_out,
                     "check IIR filter implementation");    
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }
  IR_filterTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
