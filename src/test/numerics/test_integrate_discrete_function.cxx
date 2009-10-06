//
// $Id$
//
/*!
  \file
  \ingroup test
  \brief Test integrate_discrete_function
  \author Charalampos Tsoumpas
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
 \brief tests the line_integral function 
*/

#include "stir/RunTests.h"
#include <vector>
#include "stir/numerics/integrate_discrete_function.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the integrate_discrete_function function.
*/
class integrate_discrete_functionTests : public RunTests
{
public:
  integrate_discrete_functionTests() 
  {}
  void run_tests();
};


void integrate_discrete_functionTests::run_tests()
{  
  std::cerr << "Testing integrate_discrete_function..." << std::endl;

  {  
    set_tolerance(0.000001);
    
    // Give a simple linear function. The analytical estimation of the integral is E=(tmax-tmin)*(fmax+fmin)/2
    const int Nmin=2;
    const int Nmax=333;
    
    const float tmin=12;
    const float tmax=123; 
    const float fmin=113;
    const float fmax=1113; 
    
    const float cor_integral_value=(tmax-tmin)*(fmax+fmin)*0.5;
    std::vector<float> input_vector_f(Nmax-Nmin+1), input_vector_t(Nmax-Nmin+1);

    for (int i=0;i<=Nmax-Nmin;++i)
      {
	input_vector_t[i]=tmin+i*(tmax-tmin)/(Nmax-Nmin);
	input_vector_f[i]=fmin+i*(fmax-fmin)/(Nmax-Nmin);
      }
    
    check_if_equal(integrate_discrete_function(input_vector_t,input_vector_f,0),cor_integral_value,
		   "check integrate_discrete_function implementation for linear function using rectangular approximation");  		  		  
    check_if_equal(integrate_discrete_function(input_vector_t,input_vector_f), cor_integral_value,
		   "check integrate_discrete_function implementation for linear function using trapezoidal approximation");  		  		  
  }

  // quadratic
  {
    std::vector<double> coords(100);
    std::vector<double> values(100);

    // make non-uniform sampling
    int i=0;
    for (i=0; i<=50; ++i)
      coords[i]=2*(i-50);
    for (; i<=70; ++i)
      coords[i]=coords[i-1]+1;
    for (; i<100; ++i)
      coords[i]=coords[i-1]+1.5;

    // fill values
    for (i=0; i<100; ++i)
      values[i]=2*coords[i]*coords[i]- 5*coords[i]+7;

    const double analytic_result =
      (2*coords[99]*coords[99]*coords[99]/3 - 5*coords[0]*coords[99]/2+7*coords[99]) -
      (2*coords[0]*coords[0]*coords[0]/3 - 5*coords[0]*coords[0]/2+7*coords[0]);
    double stir_result=integrate_discrete_function(coords, values,1);
    set_tolerance(.03); // expected accuracy is 3%
    check_if_equal(stir_result, analytic_result, "check integrate_discrete_function implementation for quadratic function using trapezoidal approximation");
    stir_result=integrate_discrete_function(coords, values, 0);
    set_tolerance(.03);
    check_if_equal(stir_result, analytic_result, "check integrate_discrete_function implementation for linear function using rectangular approximation"); 
  }
}

END_NAMESPACE_STIR
USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    std::cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }
  integrate_discrete_functionTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
