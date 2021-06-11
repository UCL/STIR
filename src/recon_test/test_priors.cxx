/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_test
  
  \brief Test program for stir::QuadraticPrior, RelativeDifferencePrior, and LogcoshPrior

  \par Usage

  <pre>
  test_priors [ density_filename ] 
  </pre>
  where the argument is optional. See the class documentation for more info.

  \author Kris Thielemans
  \author Robert Twyman
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/recon_buildblock/LogcoshPrior.h"
//#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/RunTests.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/num_threads.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

START_NAMESPACE_STIR


/*!
  \ingroup test
  \brief Test class for QuadraticPrior, RelativeDifferencePrior, and LogcoshPrior

  This test compares the result of GeneralisedPrior::compute_gradient()
  with a numerical gradient computed by using the 
  GeneralisedPrior::compute_value() function.
  Additionally, the Hessian's convexity is tested, via GeneralisedPrior::accumulate_Hessian_times_input(),
  by evaluating the x^T Hx > 0 constraint.

*/
class GeneralisedPriorTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  /*! This makes it possible to run the test with your own data. However, beware that
      it is very easy to set up a very long computation. 

      \todo it would be better to parse an objective function. That would allow us to set
      all parameters from the command line.
  */
  GeneralisedPriorTests(char const * const density_filename = 0);
  typedef DiscretisedDensity<3,float> target_type;
  void construct_input_data(shared_ptr<target_type>& density_sptr);

  void run_tests();
protected:
  char const * density_filename;
  shared_ptr<GeneralisedPrior<target_type> >  objective_function_sptr;

  //! run the test
  /*! Note that this function is not specific to a particular prior */
  void run_tests_for_objective_function(const std::string& test_name,
                                        GeneralisedPrior<target_type>& objective_function,
                                        shared_ptr<target_type> target_sptr);

  void test_gradient(const std::string& test_name,
                     GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                     shared_ptr<GeneralisedPriorTests::target_type> target_sptr);

  //! Test various configurations of the prior's Hessian via accumulate_Hessian_times_input()
  /*!
    Tests the concave function condition
    \f[ x^T \cdot H_{\lambda}x >= 0 \f]
    for all non-negative \c x and non-zero \c \lambda (Relative Difference Penalty conditions).
    This function constructs an array of configurations to test this condition and calls \c test_Hessian_configuration().
  */
  void test_Hessian(const std::string& test_name,
                     GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                     shared_ptr<GeneralisedPriorTests::target_type> target_sptr);

private:
  //! Hessian test for a particular configuration of the Hessian concave condition
  void test_Hessian_configuration(const std::string& test_name,
                                  GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                  shared_ptr<GeneralisedPriorTests::target_type> target_sptr,
                                  float beta,
                                  float input_multiplication, float input_addition,
                                  float current_image_multiplication, float current_image_addition);
};

GeneralisedPriorTests::
GeneralisedPriorTests(char const * const density_filename)
  : density_filename(density_filename)
{}

void
GeneralisedPriorTests::
run_tests_for_objective_function(const std::string& test_name,
                                 GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                 shared_ptr<GeneralisedPriorTests::target_type> target_sptr)
{
  std::cerr << "----- test " << test_name << '\n';
  if (!check(objective_function.set_up(target_sptr)==Succeeded::yes, "set-up of objective function"))
    return;

  std::cerr << "----- test " << test_name << "  --> Gradient\n";
  test_gradient(test_name, objective_function, target_sptr);
  std::cerr << "----- test " << test_name << "  --> Hessian-vector product (accumulate_Hessian_times_input)\n";
  test_Hessian(test_name, objective_function, target_sptr);
}


void
GeneralisedPriorTests::
test_gradient(const std::string& test_name,
              GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
              shared_ptr<GeneralisedPriorTests::target_type> target_sptr)
{
  // setup images
  target_type& target(*target_sptr);
  shared_ptr<target_type> gradient_sptr(target.get_empty_copy());
  shared_ptr<target_type> gradient_2_sptr(target.get_empty_copy());

  info("Computing gradient",3);
  objective_function.compute_gradient(*gradient_sptr, target);
  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), fabs(double(gradient_sptr->find_max()))) /1000);

  info("Computing objective function at target",3);
  const double value_at_target = objective_function.compute_value(target);
  target_type::full_iterator target_iter=target.begin_all();
  target_type::full_iterator gradient_iter=gradient_sptr->begin_all();
  target_type::full_iterator gradient_2_iter=gradient_2_sptr->begin_all();

  // setup perturbation response
  const float eps = 1e-3F;
  bool testOK = true;
  info("Computing gradient of objective function by numerical differences (this will take a while)",3);
  while(target_iter!=target.end_all())// && testOK)
  {
    const float org_image_value = *target_iter;
    *target_iter += eps;  // perturb current voxel
    const double value_at_inc = objective_function.compute_value(target);
    *target_iter = org_image_value; // restore
    const float ngradient_at_iter = static_cast<float>((value_at_inc - value_at_target)/eps);
    *gradient_2_iter = ngradient_at_iter;
    testOK = testOK && this->check_if_equal(ngradient_at_iter, *gradient_iter, "gradient");
    //for (int i=0; i<5 && target_iter!=target.end_all(); ++i)
    {
      ++gradient_2_iter; ++target_iter; ++ gradient_iter;
    }
  }
  if (!testOK)
  {
    info("Writing diagnostic files gradient" + test_name + ".hv, numerical_gradient" + test_name + ".hv");
    write_to_file("gradient" + test_name + ".hv", *gradient_sptr);
    write_to_file("numerical_gradient" + test_name + ".hv", *gradient_2_sptr);
  }
}

void
GeneralisedPriorTests::
test_Hessian(const std::string& test_name,
              GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
              shared_ptr<GeneralisedPriorTests::target_type> target_sptr)
{
  /// Construct configurations
  float beta_array[] = {0.01, 1, 100};  // Penalty strength should only affect scale
  // Modifications to the input image
  float input_multiplication_array[] = {-100, -1, 0.01, 1, 100}; // Test negative, small and large values
  float input_addition_array[] = {-10, -1, -0.5, 0.0, 1, 10};
  // Modifications to the current image (Hessian computation)
  float current_image_multiplication_array[] = {0.01, 1, 100};
  float current_image_addition_array[] = {0.0, 0.5, 1, 10}; // RDP has constraint that current_image is non-negative

  float initial_beta = objective_function.get_penalisation_factor();
  for (float beta : beta_array) {
    for (float input_multiplication : input_multiplication_array) {
      for (float input_addition : input_addition_array) {
        for (float current_image_multiplication : current_image_multiplication_array) {
          for (float current_image_addition : current_image_addition_array) {
            test_Hessian_configuration(test_name, objective_function, target_sptr,
                                       beta,
                                       input_multiplication, input_addition,
                                       current_image_multiplication, current_image_addition);
          }
        }
      }
    }
  }
  /// Reset beta to original value
  objective_function.set_penalisation_factor(initial_beta);
}

void
GeneralisedPriorTests::
test_Hessian_configuration(const std::string& test_name,
                           GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                           shared_ptr<GeneralisedPriorTests::target_type> target_sptr,
                           const float beta,
                           const float input_multiplication, const float input_addition,
                           const float current_image_multiplication, const float current_image_addition)
{
  /// setup images
  target_type& target(*target_sptr);
  shared_ptr<target_type> output(target.get_empty_copy());
  shared_ptr<target_type> current_image(target.get_empty_copy());
  shared_ptr<target_type> input(target.get_empty_copy());
  objective_function.set_penalisation_factor(beta);
  {
    /// Construct an current_image & input with various values that are variations of the target
    target_type::full_iterator input_iter = input->begin_all();
    target_type::full_iterator current_image_iter = current_image->begin_all();
    target_type::full_iterator target_iter = target.begin_all();
    while (input_iter != input->end_all())// && testOK)
    {
      *input_iter = input_multiplication * *target_iter + input_addition;
      *current_image_iter = current_image_multiplication * *target_iter + current_image_addition;
      ++input_iter; ++target_iter; ++current_image_iter;
    }
  }

  /// Compute H x
  objective_function.accumulate_Hessian_times_input(*output, *current_image, *input);

  /// Compute x \cdot (H x)
  float my_sum = 0.0;
  my_sum = std::inner_product(input->begin_all(), input->end_all(), output->begin_all(), my_sum);

  // test for a CONVEX function
  if (this->check_if_less(0, my_sum)) {
//    info("PASS: Computation of x^T H x = " + std::to_string(my_sum) + " > 0 and is therefore convex");
  } else {
    // print to console the FAILED configuration
    info("FAIL: Computation of x^T H x = " + std::to_string(my_sum) + " < 0 and is therefore NOT convex" +
         "\ntest_name=" + test_name +
         "\nbeta=" + std::to_string(beta) +
         "\ninput_multiplication=" + std::to_string(input_multiplication) +
         "\ninput_addition=" + std::to_string(input_addition) +
         "\ncurrent_image_multiplication=" + std::to_string(current_image_multiplication) +
         "\ncurrent_image_addition=" + std::to_string(current_image_addition) +
         "\n >input image max=" + std::to_string(input->find_max()) +
         "\n >input image min=" + std::to_string(input->find_min()) +
         "\n >target image max=" + std::to_string(target.find_max()) +
         "\n >target image min=" + std::to_string(target.find_min()));
  }
}

void
GeneralisedPriorTests::
construct_input_data(shared_ptr<target_type>& density_sptr)
{
  if (this->density_filename == 0)
    {
      // construct a small image with random voxel values between 0 and 1

      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      exam_info_sptr->imaging_modality = ImagingModality::PT;
      CartesianCoordinate3D<float> origin (0,0,0);
      CartesianCoordinate3D<float> voxel_size(2.F,3.F,3.F);
      
      density_sptr.reset(new VoxelsOnCartesianGrid<float>(exam_info_sptr,
                                                          IndexRange<3>(make_coordinate(10,9,8)),
                                                          origin, voxel_size));
      // fill with random numbers between 0 and 1
      typedef boost::mt19937 base_generator_type;
      // initialize by reproducible seed
      static base_generator_type generator(boost::uint32_t(42));
      static boost::uniform_01<base_generator_type> random01(generator);
      for (target_type::full_iterator iter=density_sptr->begin_all(); iter!=density_sptr->end_all(); ++iter)
        *iter = static_cast<float>(random01());

    }
  else
    {
      // load image from file
      shared_ptr<target_type> aptr(read_from_file<target_type>(this->density_filename));
      density_sptr = aptr;
    }
    return;
}

void
GeneralisedPriorTests::
run_tests()
{
  shared_ptr<target_type> density_sptr;
  construct_input_data(density_sptr);

  std::cerr << "Tests for QuadraticPrior\n";
  {
    QuadraticPrior<float> objective_function(false, 1.F);
    this->run_tests_for_objective_function("Quadratic_no_kappa", objective_function, density_sptr);
  }
  std::cerr << "Tests for Relative Difference Prior\n";
  {
    // gamma is default and epsilon is off
    RelativeDifferencePrior<float> objective_function(false, 1.F, 2.F, 0.F);
    this->run_tests_for_objective_function("RDP_no_kappa", objective_function, density_sptr);
  }
  // Disabled PLS due to known issue
//  std::cerr << "Tests for PLSPrior\n";
//  {
//    PLSPrior<float> objective_function(false, 1.F);
//    shared_ptr<DiscretisedDensity<3,float> > anatomical_image_sptr(density_sptr->get_empty_copy());
//    anatomical_image_sptr->fill(1.F);
//    objective_function.set_anatomical_image_sptr(anatomical_image_sptr);
//    this->run_tests_for_objective_function("PLS_no_kappa_flat_anatomical", objective_function, density_sptr);
//  }
  std::cerr << "Tests for Logcosh Prior\n";
  {
    // scalar is off
    LogcoshPrior<float> objective_function(false, 1.F, 1.F);
    this->run_tests_for_objective_function("Logcosh_no_kappa", objective_function, density_sptr);
  }
} 

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  set_default_num_threads();

  GeneralisedPriorTests tests(argc>1? argv[1] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
