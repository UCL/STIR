//
// $Id$
//
/*!
  \file 
  \ingroup test
  \brief tests the BSplines_1st_der_weight function

  \author Charalampos Tsoumpas
  \author Kris Thielemans

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
#include "local/stir/BSplines.h"
//#include "local/stir/BSplines_coef.h"
#include <vector>
#include <algorithm>


#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the BSplines_coef function.
*/
class BSplines_1st_der_weightTests : public RunTests
{
public:
  BSplines_1st_der_weightTests() 
  {}
  void run_tests();
private:
  //istream& in;
};


void BSplines_1st_der_weightTests::run_tests()
{  
  cerr << "Testing BSplines_1st_der_weight function..." << endl;

  set_tolerance(0.00001);
 
  typedef double elemT;
  std::vector<elemT> BSplines_1st_der_weight_STIR_vector, BSplines_1st_der_weight_correct_vector;
  
  BSplines_1st_der_weight_STIR_vector.push_back(BSplines_1st_der_weight(0.));

  for(elemT i=0.3; i<=3 ;++i)
	  BSplines_1st_der_weight_STIR_vector.push_back(BSplines_1st_der_weight(i));

	  BSplines_1st_der_weight_correct_vector.push_back(0.); //1
	  BSplines_1st_der_weight_correct_vector.push_back(-0.465); //2
	  BSplines_1st_der_weight_correct_vector.push_back(-0.245); //3
	  BSplines_1st_der_weight_correct_vector.push_back(0.); //4
	  
	  std::vector<elemT>:: iterator cur_iter_stir_out= BSplines_1st_der_weight_STIR_vector.begin()
		  , 	  cur_iter_test= BSplines_1st_der_weight_correct_vector.begin()		  ;
	  for (; cur_iter_stir_out!=BSplines_1st_der_weight_STIR_vector.end() &&
		  cur_iter_test!=BSplines_1st_der_weight_correct_vector.end();	  
		    ++cur_iter_stir_out, ++cur_iter_test)			  
				check_if_equal(*cur_iter_stir_out, *cur_iter_test,
				"check BSplines_1st_der_weight implementation");    		  		  		  
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
  BSplines_1st_der_weightTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
