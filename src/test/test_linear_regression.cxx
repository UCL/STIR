//
//
/*!

  \file
  \ingroup test

  \brief A simple programme to test the stir::linear_regression function.

  \author Kris Thielemans
  \author PARAPET project



  
  To run the test, you should use a command line argument with the name of a file.
  This should contain a number of test cases for the fit.
  See linear_regressionTests for file contents.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
  
#include "stir/linear_regression.h"
#include "stir/RunTests.h"
#include "stir/ArrayFunction.h"

#include <fstream>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the linear_regression function.

  The class reads input from a stream, whose contents should be as follows:

    1 line with some general text<br>
    1 line with some text for the next test case<br>
    list_of_numbers<br>
    1 line with some text for the next test case<br>
    list_of_numbers<br>
    ...

    
    where list_of_numbers is the following list of numbers 
    (white space is ignored)

    number_of_points<br>
    coordinates<br>
    data<br>
    weights<br>
    expected_constant  expected_scale  <br>
    expected_chi_square <br> 
    expected_variance_of_constant  expected_variance_of_scale  <br>
    expected_covariance_of_constant_with_scale<br>

*/
class linear_regressionTests : public RunTests
{
public:
  linear_regressionTests(istream& in) 
    : in(in)
  {}

  void run_tests();
private:
  istream& in;
};


void linear_regressionTests::run_tests()
{  
  cerr << "Testing linear_regression function..." << endl;

  char text[200];
  in.get(text,200);
  cerr << text << endl;
  int test_case = 0;
  
  while (in)
  {
    // first get rid of EOL
    in.getline(text,200);
    // now get real text
    in.get(text,200);
    test_case ++;
    
    int size = 0;
    in >> size;
    if (size<=0)
      break;

    cerr << text << endl;
    Array<1,float> coordinates(size);
    Array<1,float> measured_data(size);
    Array<1,float> weights(size);
    for (int i=0; i<size; i++)
      in >> coordinates[i];
    for (int i=0; i<size; i++)
      in >> measured_data[i];
    for (int i=0; i<size; i++)
      in >> weights[i];
    
    double expected_scale;
    double expected_constant;
    double expected_variance_of_scale;
    double expected_variance_of_constant;
    double expected_covariance_of_constant_with_scale;
    double expected_chi_square;
    
    in >> expected_constant >> expected_scale 
      >> expected_chi_square 
      >> expected_variance_of_constant >> expected_variance_of_scale 
      >> expected_covariance_of_constant_with_scale;
    
    
    double scale=0;
    double constant=0;
    double variance_of_scale=0;
    double variance_of_constant=0;
    double covariance_of_constant_with_scale=0;
    double chi_square = 0;
    
    linear_regression(
      constant, scale,
      chi_square,
      variance_of_constant,
      variance_of_scale,
      covariance_of_constant_with_scale,
      measured_data,
      coordinates,
      weights);
    
    check_if_equal(expected_constant, constant, 
		    "for parameter constant, should be equal");
    check_if_equal(expected_scale, scale, 
		    "for parameter scale, should be equal");

    // KT 24/01/2001 changed tolerance for this comparisons
    /* chi_square is computed as a 

        chi^2 = weights.(data - (constant + scale coordinates))^2
 
       There's a subtraction of  nearly matching floating
       point numbers. This means there's a lot of potential for 
       numerical error.
       Float computations have a precision of about 10^-5. Which
       means you can really trust numbers upto 10^-5 times some value
       related to the size of the numbers you're adding/subtracing.

       So, we're really computing
	chi^2 +- 10^-5*(weights.data)
    */
    double error_on_chi_square = 0;
    {
      for (int i=0; i<size; ++i)	
	error_on_chi_square += fabs(weights[i]*measured_data[i]);
      error_on_chi_square *= 10.E-5;
    }
    {
      const double old_tolerance = get_tolerance();
      set_tolerance(error_on_chi_square);
      check_if_equal(expected_chi_square, chi_square, 
		     "for parameter chi_square, should be equal");

      // first check if chi_square is really 0 (upto rounding errors)
      if (fabs(expected_chi_square) < error_on_chi_square)
	set_tolerance(old_tolerance);

      // next variances are proportional to chi_square
      if (expected_chi_square != 0)
	set_tolerance(fabs(expected_variance_of_constant/expected_chi_square*error_on_chi_square));
      check_if_equal(expected_variance_of_constant, variance_of_constant, 
		     "for parameter variance_of_constant, should be equal");
      if (expected_chi_square != 0)
	set_tolerance(fabs(expected_variance_of_scale/expected_chi_square*error_on_chi_square));
      check_if_equal(expected_variance_of_scale, variance_of_scale, 
		     "for parameter variance_of_scale, should be equal");
      if (expected_chi_square != 0)
	set_tolerance(fabs(expected_covariance_of_constant_with_scale/expected_chi_square*error_on_chi_square));
      /*
      std::cerr << "chi_square " << chi_square
		<< "\nexpected_chi_square " << expected_chi_square
		<< "\error on chi_square " << error_on_chi_square
		<< "\nRelative error on chi_square " << 1/expected_chi_square*error_on_chi_square
		<< "\n cov_const_scale_tolerance " << get_tolerance()
		<<std::endl;
      */
      check_if_equal(expected_covariance_of_constant_with_scale, 
		     covariance_of_constant_with_scale, 
		     "for parameter covariance_of_constant_with_scale should be equal");  
      set_tolerance(old_tolerance);
    }
  }
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " filename\n"
         << "See source file for the format of this file.\n\n";
    return EXIT_FAILURE;
  }


  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0] 
         << ": Error opening input file " << argv[1] << "\nExiting.\n";

    return EXIT_FAILURE;
  }

  linear_regressionTests tests(in);
  tests.run_tests();
  return tests.main_return_value();
}
