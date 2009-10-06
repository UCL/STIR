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

  
  set_tolerance(0.000001);
  //  float STIR_rect_result;

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
