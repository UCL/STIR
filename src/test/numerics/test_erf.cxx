//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup numerics_test
  \brief tests the error function stir::erf and its complementary

  \author Charalampos Tsoumpas

*/
  
#include "stir/RunTests.h"
#include "stir/numerics/erf.h"
#include "stir/numerics/FastErf.h"
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
  void run_tests() override;

  /*!\brief Tests STIR's erf(x) function against known values */
  void test_stir_erf();

  /*!\brief Tests the FastErf object and its interpolation methods.
   * This test will construct a FastErf object and compute a range of erf values and compare to erf(x)
    */
  void test_FastErf();

private:
  //! Executes all the FastErf interpolation methods for a given xp
  void actual_test_FastErf(const float xp);

  FastErf e;
};


void erfTests::run_tests()
{
  std::cerr << "Testing Error Functions..." << std::endl;
  test_stir_erf();
  test_FastErf();
}

void erfTests::test_stir_erf()
{
  std::cerr << "  Testing stir error functions..." << std::endl;

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

void
erfTests::test_FastErf()
{
  std::cerr << "  Testing stir FastErf ..." << std::endl;
  set_tolerance(0.0001);

  e.set_num_samples(200000);
  e.set_up();

  const float upper_samle_limit = 2 * e.get_maximum_sample_value() + 1;
  const float lower_samle_limit = -(upper_samle_limit);
  double sample_period = _PI/ 10000;  // Needed a number that wasn't regular and this worked...
  // Test the FastErf interpolations 2* beyond the _maximum_sample_value.
  // The while (-_maximum_sample_value > xp) or (_maximum_sample_value < xp),
  // xp is clamped to -_maximum_sample_value or _maximum_sample_value
  for (double xp = lower_samle_limit; xp < upper_samle_limit; xp += sample_period)
    {
      this->actual_test_FastErf(xp);
      if (!this->everything_ok)
        break;
    }

  // Test cases where x is just smaller or larger than the lower or upper limits of FastErf
  // This is an additional sanity check ensure there are no rounding or out of limit errors.
  const float epsilon = sample_period/100;
  std::vector<float> extremity_test_xps(4);
  extremity_test_xps[0] = e.get_maximum_sample_value() + epsilon; // above max
  extremity_test_xps[1] = e.get_maximum_sample_value() - epsilon; // under max
  extremity_test_xps[2] = -e.get_maximum_sample_value() + epsilon;  // above min
  extremity_test_xps[3] = -e.get_maximum_sample_value() - epsilon;  // under min

  for (float xp : extremity_test_xps)
    {
      this->actual_test_FastErf(xp);
      if (!this->everything_ok)
        break;
    }
}



void erfTests::actual_test_FastErf(const float xp)
{
  //BSPlines
  check_if_equal(e.get_erf_BSplines_interpolation(xp), erf(xp));
  if (!this->is_everything_ok()){
    std::cerr << "xp = " << xp
              << "\tFastErf.get_erf_BSplines_interpolation(xp) = " << e.get_erf_BSplines_interpolation(xp)
              << "\terf(xp) = " << erf(xp) << "\n";
  }
  // Linear
  check_if_equal(e.get_erf_linear_interpolation(xp), erf(xp));
  if (!this->is_everything_ok()){
    std::cerr << "linear xp = " << xp
              << "\tFastErf.get_erf_linear_interpolation(xp) = " << e.get_erf_linear_interpolation(xp)
              << "\terf(xp) = " << erf(xp) << "\n";
  }

  //NN
  check_if_equal(e.get_erf_nearest_neighbour_interpolation(xp), erf(xp));
  if (!this->is_everything_ok()){
    std::cerr << "NN xp = " << xp
              << "\tFastErf.get_erf_nearest_neighbour_interpolation(xp) = " << e.get_erf_nearest_neighbour_interpolation(xp)
              << "\terf(xp) = " << erf(xp) << "\n";
  }

  // Operator () - This acts as a wrapper for e.get_erf_linear_interpolation(xp)
  check_if_equal(e(xp), erf(xp));
  if (!this->is_everything_ok()){
    std::cerr << "NN xp = " << xp
              << "\tFastErf(xp) = " << e(xp)
              << "\terf(xp) = " << erf(xp) << "\n";
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
