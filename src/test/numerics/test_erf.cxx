//
// $Id$
//
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
  \ingroup numerics_test
  \brief tests the error function stir::erf and its complementary

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
  
#include "stir/RunTests.h"
#include "stir/numerics/erf.h"
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup numerics_test
  \brief A simple class to test the erf and erfc functions.
*/
class erfTests : public RunTests
{
public:
  erfTests() 
  {}
  void run_tests();
private:
  //istream& in;
};


void erfTests::run_tests()
{  
  std::cerr << "Testing Error Functions..." << std::endl;

  set_tolerance(0.000000000000001);
 
  //std::vector<float> input, mathematica_results, STIR_results;
	  
  {   
	
    std::vector<double> input_vector(10), 
      output_correct_vector(10), output_correct_vector_c(10),
      STIR_vector(10), STIR_vector_c(10);
	
    output_correct_vector[0] = 0.0000000000000000000000000000000;
    output_correct_vector[1] = 0.0112834155558496169159095235481;
    output_correct_vector[2] = 0.0225645746918449442243658616474;
    output_correct_vector[3] = 0.0338412223417354333022166542569;
    output_correct_vector[4] = 0.0451111061451247520897490647292;
    output_correct_vector[5] = 0.0563719777970166238312711126020;
    output_correct_vector[6] = 0.0676215943933084420794314523912;
    output_correct_vector[7] = 0.0788577197708907433569970386680;
    output_correct_vector[8] = 0.0900781258410181607233921876161;
    output_correct_vector[9] = 0.101280593914626883352498163244;		  

    output_correct_vector_c[0] = 1.00000000000000000000000000000;
    output_correct_vector_c[1] = 0.988716584444150383084090476452;
    output_correct_vector_c[2] = 0.977435425308155055775634138353;
    output_correct_vector_c[3] = 0.966158777658264566697783345743;
    output_correct_vector_c[4] = 0.954888893854875247910250935271;
    output_correct_vector_c[5] = 0.943628022202983376168728887398;
    output_correct_vector_c[6] = 0.932378405606691557920568547609;
    output_correct_vector_c[7] = 0.921142280229109256643002961332;
    output_correct_vector_c[8] = 0.909921874158981839276607812384;
    output_correct_vector_c[9] = 0.898719406085373116647501836756;		  


    for (int i=0 ; i<10 ; ++i)		  
      {
        input_vector[i] = 0.01*(static_cast<double>(i));
        STIR_vector[i] = erf(input_vector[i]);
        STIR_vector_c[i] = erfc(input_vector[i]);			 
      }
    std::vector<double>:: iterator cur_iter_cor = output_correct_vector.begin(), 
      cur_iter_cor_c = output_correct_vector_c.begin(), 
      cur_iter_STIR = STIR_vector.begin(),
      cur_iter_STIR_c = STIR_vector_c.begin();		  
		  		  
    for (cur_iter_STIR = STIR_vector.begin();
         cur_iter_STIR!=STIR_vector.end() && cur_iter_cor!=output_correct_vector.end();
         ++cur_iter_STIR,  ++cur_iter_cor)
      check_if_equal(*cur_iter_cor, *cur_iter_STIR,
                     "check erf() implementation");  		  		  
    for (;
         cur_iter_STIR_c!=STIR_vector_c.end() && cur_iter_cor_c!=output_correct_vector_c.end();
         ++cur_iter_STIR_c,  ++cur_iter_cor_c)
      check_if_equal(*cur_iter_cor_c, *cur_iter_STIR_c,
                     "check erfc() implementation");  		  		  

    for (cur_iter_STIR_c = STIR_vector_c.begin(), cur_iter_STIR = STIR_vector.begin();
         cur_iter_STIR!=STIR_vector.end() &&  
           cur_iter_STIR_c!=STIR_vector_c.end();
         ++cur_iter_STIR,  ++cur_iter_STIR_c)
      check_if_equal(1.0, (*cur_iter_STIR_c) + (*cur_iter_STIR),
                     "check erfc() and erf() results");    		  
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
  erfTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
