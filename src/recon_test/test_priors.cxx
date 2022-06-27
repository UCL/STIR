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
#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/RunTests.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Verbosity.h"
#include "stir/Succeeded.h"
#include "stir/num_threads.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

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
  explicit GeneralisedPriorTests(char const * density_filename = nullptr);
  typedef DiscretisedDensity<3,float> target_type;
  void construct_input_data(shared_ptr<target_type>& density_sptr);

  void run_tests() override;

  //! Set methods that control which tests are run.
  void configure_prior_tests(bool gradient, bool Hessian_convexity, bool Hessian_numerical);

protected:
  char const * density_filename;
  shared_ptr<GeneralisedPrior<target_type> >  objective_function_sptr;

  //! run the test
  /*! Note that this function is not specific to a particular prior */
  void run_tests_for_objective_function(const std::string& test_name,
                                        GeneralisedPrior<target_type>& objective_function,
                                        const shared_ptr<target_type>& target_sptr);

  //! Tests the prior's gradient by comparing to the numerical gradient computed using perturbation response.
  void test_gradient(const std::string& test_name,
                     GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                     const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr);

  //! Test various configurations of the Hessian of the prior via accumulate_Hessian_times_input() for convexity
  /*!
    Tests the convexity condition:
    \f[ x^T \cdot H_{\lambda}x >= 0 \f]
    for all non-negative \c x and non-zero \c \lambda (Relative Difference Prior conditions).
    This function constructs an array of configurations to test this condition and calls
    \c test_Hessian_convexity_configuration().
  */
  void test_Hessian_convexity(const std::string& test_name,
                              GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                              const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr);

  //! Tests the compute_Hessian method implemented into convex priors
  /*! Performs a perturbation response using compute_gradient to determine if the compute_Hessian (for a single densel)
      is within tolerance.
  */
  void test_Hessian_against_numerical(const std::string& test_name,
                                      GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                      const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr);

private:
  //! Hessian test for a particular configuration of the Hessian concave condition
  bool test_Hessian_convexity_configuration(const std::string& test_name,
                                            GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                            const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr,
                                            float beta,
                                            float input_multiplication, float input_addition,
                                            float current_image_multiplication, float current_image_addition);

  //! Variables to control which tests are run, see the set methods
  //@{
  bool do_test_gradient = false;
  bool do_test_Hessian_convexity = false;
  bool do_test_Hessian_against_numerical = false;
  //@}
};

GeneralisedPriorTests::
GeneralisedPriorTests(char const * const density_filename)
  : density_filename(density_filename)
{}

void
GeneralisedPriorTests::configure_prior_tests(const bool gradient, const bool Hessian_convexity, const bool Hessian_numerical)
{
  do_test_gradient = gradient;
  do_test_Hessian_convexity = Hessian_convexity;
  do_test_Hessian_against_numerical = Hessian_numerical;
}

void
GeneralisedPriorTests::
run_tests_for_objective_function(const std::string& test_name,
                                 GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                 const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  std::cerr << "----- test " << test_name << '\n';
  if (!check(objective_function.set_up(target_sptr)==Succeeded::yes, "set-up of objective function"))
    return;

  if (do_test_gradient)
  {
    std::cerr << "----- test " << test_name << "  --> Gradient\n";
    test_gradient(test_name, objective_function, target_sptr);
  }

  if (do_test_Hessian_convexity)
  {
    std::cerr << "----- test " << test_name << "  --> Hessian-vector product for convexity\n";
    test_Hessian_convexity(test_name, objective_function, target_sptr);
  }

  if (do_test_Hessian_against_numerical)
  {
    std::cerr << "----- test " << test_name << "  --> Hessian against numerical\n";
    test_Hessian_against_numerical(test_name, objective_function, target_sptr);
  }
}


void
GeneralisedPriorTests::
test_gradient(const std::string& test_name,
              GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
              const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  // setup images
  target_type& target(*target_sptr);
  shared_ptr<target_type> gradient_sptr(target.get_empty_copy());
  shared_ptr<target_type> gradient_2_sptr(target.get_empty_copy());

  info("Computing gradient",3);
  const int verbosity_default = Verbosity::get();
  Verbosity::set(0);
  objective_function.compute_gradient(*gradient_sptr, target);
  Verbosity::set(verbosity_default);
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
    const auto ngradient_at_iter = static_cast<float>((value_at_inc - value_at_target)/eps);
    *gradient_2_iter = ngradient_at_iter;
    testOK = testOK && this->check_if_equal(ngradient_at_iter, *gradient_iter, "gradient");
    //for (int i=0; i<5 && target_iter!=target.end_all(); ++i)
    {
      ++gradient_2_iter; ++target_iter; ++ gradient_iter;
    }
  }
  if (!testOK)
  {
    std::cerr << "Numerical gradient test failed with for " + test_name + " prior\n";
    info("Writing diagnostic files gradient" + test_name + ".hv, numerical_gradient" + test_name + ".hv");
    write_to_file("gradient" + test_name + ".hv", *gradient_sptr);
    write_to_file("numerical_gradient" + test_name + ".hv", *gradient_2_sptr);
  }
}

void
GeneralisedPriorTests::
test_Hessian_convexity(const std::string& test_name,
                       GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                       const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  if (!objective_function.is_convex())
    return;
  /// Construct configurations
  float beta_array[] = {0.01, 1, 100};  // Penalty strength should only affect scale
  // Modifications to the input image
  float input_multiplication_array[] = {-100, -1, 0.01, 1, 100}; // Test negative, small and large values
  float input_addition_array[] = {-10, -1, -0.5, 0.0, 1, 10};
  // Modifications to the current image (Hessian computation)
  float current_image_multiplication_array[] = {0.01, 1, 100};
  float current_image_addition_array[] = {0.0, 0.5, 1, 10}; // RDP has constraint that current_image is non-negative

  bool testOK = true;
  float initial_beta = objective_function.get_penalisation_factor();
  for (float beta : beta_array)
    for (float input_multiplication : input_multiplication_array)
      for (float input_addition : input_addition_array)
        for (float current_image_multiplication : current_image_multiplication_array)
          for (float current_image_addition : current_image_addition_array) {
            if (testOK)  // only compute configuration if testOK from previous tests
              testOK = test_Hessian_convexity_configuration(test_name, objective_function, target_sptr,
                                                            beta,
                                                            input_multiplication, input_addition,
                                                            current_image_multiplication, current_image_addition);
          }
  /// Reset beta to original value
  objective_function.set_penalisation_factor(initial_beta);
}

bool
GeneralisedPriorTests::
test_Hessian_convexity_configuration(const std::string& test_name,
                                     GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                     const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr,
                                     const float beta,
                                     const float input_multiplication, const float input_addition,
                                     const float current_image_multiplication, const float current_image_addition)
{
  /// setup targets
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
    while (input_iter != input->end_all())
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
    return true;
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
    return false;
  }
}


void
GeneralisedPriorTests::
test_Hessian_against_numerical(const std::string &test_name,
                               GeneralisedPrior<GeneralisedPriorTests::target_type> &objective_function,
                               const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  if (!objective_function.is_convex())
    return;

  /// Setup
  const float eps = 1e-3F;
  bool testOK = true;
  const int verbosity_default = Verbosity::get();

  // setup images
  target_type& input(*target_sptr->get_empty_copy());
  input += *target_sptr;  // make input have same values as target_sptr
  shared_ptr<target_type> gradient_sptr(target_sptr->get_empty_copy());
  shared_ptr<target_type> pert_grad_and_numerical_Hessian_sptr(target_sptr->get_empty_copy());
  shared_ptr<target_type> Hessian_sptr(target_sptr->get_empty_copy());

  Verbosity::set(0);
  objective_function.compute_gradient(*gradient_sptr, input);
  Verbosity::set(verbosity_default);
//  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), fabs(double(gradient_sptr->find_max()))) /10);

  // Setup coordinates (z,y,x) for perturbation test (Hessian will also be computed w.r.t this voxel, j)
  BasicCoordinate<3, int> perturbation_coords;

  // Get min/max indices
  const int min_z = input.get_min_index();
  const int max_z = input.get_max_index();
  const int min_y = input[min_z].get_min_index();
  const int max_y = input[min_z].get_max_index();
  const int min_x = input[min_z][min_y].get_min_index();
  const int max_x = input[min_z][min_y].get_max_index();

  // Loop over each voxel j in the input and check perturbation response.
  for (int z=min_z;z<= max_z;z++)
    for (int y=min_y;y<= max_y;y++)
      for (int x=min_x;x<= max_x;x++)
        if (testOK)
        {
          perturbation_coords[1] = z; perturbation_coords[2] = y; perturbation_coords[3] = x;

          //  Compute H(x)_j (row of the Hessian at the jth voxel)
          objective_function.compute_Hessian(*Hessian_sptr, perturbation_coords, input);
          this->set_tolerance(std::max(fabs(double(Hessian_sptr->find_min())), fabs(double(Hessian_sptr->find_max()))) /500);

          // Compute g(x + eps)
          Verbosity::set(0);
          // Perturb target at jth voxel, compute perturbed gradient, and reset voxel to original value
          float perturbed_voxels_original_value = input[perturbation_coords[1]][perturbation_coords[2]][perturbation_coords[3]];
          input[perturbation_coords[1]][perturbation_coords[2]][perturbation_coords[3]] += eps;
          objective_function.compute_gradient(*pert_grad_and_numerical_Hessian_sptr, input);
          input[perturbation_coords[1]][perturbation_coords[2]][perturbation_coords[3]] = perturbed_voxels_original_value;

          // Now compute the numerical-Hessian = (g(x+eps) - g(x))/eps
          *pert_grad_and_numerical_Hessian_sptr -= *gradient_sptr;
          *pert_grad_and_numerical_Hessian_sptr /= eps;

          Verbosity::set(verbosity_default);
          // Test if pert_grad_and_numerical_Hessian_sptr is all zeros.
          // This can happen if the eps is too small. This is a quick test that allows for easier debugging.
          if (pert_grad_and_numerical_Hessian_sptr->sum_positive() == 0.0 && Hessian_sptr->sum_positive() > 0.0)
          {
            this->everything_ok = false;
            testOK = false;
            info("test_Hessian_against_numerical: failed because all values are 0 in numerical Hessian");
          }

          // Loop over each of the voxels and compare the numerical-Hessian with Hessian
          target_type::full_iterator numerical_Hessian_iter = pert_grad_and_numerical_Hessian_sptr->begin_all();
          target_type::full_iterator Hessian_iter = Hessian_sptr->begin_all();
          while(numerical_Hessian_iter != pert_grad_and_numerical_Hessian_sptr->end_all())
          {
            testOK = testOK && this->check_if_equal(*Hessian_iter, *numerical_Hessian_iter, "Hessian");
            ++numerical_Hessian_iter; ++ Hessian_iter;
          }

          if (!testOK)
          {
            // Output volumes for debug
            std::cerr << "Numerical-Hessian test failed with for " + test_name + " prior\n";
            info("Writing diagnostic files `Hessian_" + test_name + ".hv` and `numerical_Hessian_" + test_name + ".hv`");
            write_to_file("Hessian_" + test_name + ".hv", *Hessian_sptr);
            write_to_file("numerical_Hessian_" + test_name + ".hv", *pert_grad_and_numerical_Hessian_sptr);
            write_to_file("input_" + test_name + ".hv", input);
          }
        }
}

void
GeneralisedPriorTests::
construct_input_data(shared_ptr<target_type>& density_sptr)
{
  if (this->density_filename == nullptr)
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
}

void
GeneralisedPriorTests::
run_tests()
{
  shared_ptr<target_type> density_sptr;
  construct_input_data(density_sptr);

  std::cerr << "\n\nTests for QuadraticPrior\n";
  {
    QuadraticPrior<float> objective_function(false, 1.F);
    this->configure_prior_tests(true, true, true);
    this->run_tests_for_objective_function("Quadratic_no_kappa", objective_function, density_sptr);
  }
  std::cerr << "\n\nTests for Relative Difference Prior with epsilon = 0\n";
  {
    // gamma is default and epsilon is 0.0
    RelativeDifferencePrior<float> objective_function(false, 1.F, 2.F, 0.F);
    this->configure_prior_tests(true, true, false); // RDP, with epsilon = 0.0, will fail the numerical Hessian test
    this->run_tests_for_objective_function("RDP_no_kappa_no_eps", objective_function, density_sptr);
  }
  std::cerr << "\n\nTests for Relative Difference Prior with epsilon = 0.1\n";
  {
    // gamma is default and epsilon is "small"
    RelativeDifferencePrior<float> objective_function(false, 1.F, 2.F, 0.1F);
    this->configure_prior_tests(true, true, true); // With a large enough epsilon the RDP Hessian numerical test will pass
    this->run_tests_for_objective_function("RDP_no_kappa_with_eps", objective_function, density_sptr);
  }

  std::cerr << "\n\nTests for PLSPrior\n";
  {
    PLSPrior<float> objective_function(false, 1.F);
    shared_ptr<DiscretisedDensity<3,float> > anatomical_image_sptr(density_sptr->get_empty_copy());
    anatomical_image_sptr->fill(1.F);
    objective_function.set_anatomical_image_sptr(anatomical_image_sptr);
    // Disabled PLS due to known issue
    this->configure_prior_tests(false, false, false);
    this->run_tests_for_objective_function("PLS_no_kappa_flat_anatomical", objective_function, density_sptr);
  }
  std::cerr << "\n\nTests for Logcosh Prior\n";
  {
    // scalar is off
    LogcoshPrior<float> objective_function(false, 1.F, 1.F);
    this->configure_prior_tests(true, true, true);
    this->run_tests_for_objective_function("Logcosh_no_kappa", objective_function, density_sptr);
  }
} 

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  set_default_num_threads();

  GeneralisedPriorTests tests(argc>1? argv[1] : nullptr);
  tests.run_tests();
  return tests.main_return_value();
}
