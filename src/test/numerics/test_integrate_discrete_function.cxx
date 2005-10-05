//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Integrating plasma data
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
#include "local/stir/numerics/linear_integral.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the linear_integral function.
*/
class linear_integralTests : public RunTests
{
public:
  linear_integralTests() 
  {}
  void run_tests();
};


void linear_integralTests::run_tests()
{  
  std::cerr << "Testing Linear Integral Functions..." << std::endl;

  set_tolerance(0.000001);
  std::vector<float> input_vector_f(10), input_vector_t(10);
  //  float integral_result, STIR_rect_result;

  for (int i=0;i<10;++i)
    {
       input_vector_f[i]=i;
       input_vector_t[i]=i+1;
    }
    
  check_if_equal(linear_integral(input_vector_f,input_vector_t,0), 36.F,
				  "check linear_integral implementation");  		  		  
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
  linear_integralTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
