/*
    Copyright (C) 2020-2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_test

  \brief Test program for stir::QuadraticPrior, stir::RelativeDifferencePrior, and stir::LogcoshPrior

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
#ifdef STIR_WITH_CUDA
#  include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#endif
#include "stir/recon_buildblock/LogcoshPrior.h"
#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/recon_buildblock/test/ObjectiveFunctionTests.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Verbosity.h"
#include "stir/Succeeded.h"
#include "stir/num_threads.h"
#include "stir/numerics/norm.h"
#include "stir/SeparableGaussianImageFilter.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for QuadraticPrior, RelativeDifferencePrior, CudaRelativeDifferencePrior and LogcoshPrior

  This test compares the result of GeneralisedPrior::compute_gradient()
  with a numerical gradient computed by using the
  GeneralisedPrior::compute_value() function.
  Additionally, the Hessian's convexity is tested, via GeneralisedPrior::accumulate_Hessian_times_input(),
  by evaluating the x^T Hx > 0 constraint.

*/
class GeneralisedPriorTests
    : public ObjectiveFunctionTests<GeneralisedPrior<DiscretisedDensity<3, float>>, DiscretisedDensity<3, float>>
{
public:
  //! Constructor that can take some input data to run the test with
  /*! This makes it possible to run the test with your own data. However, beware that
      it is very easy to set up a very long computation.

      \todo it would be better to parse an objective function. That would allow us to set
      all parameters from the command line.
  */
  explicit GeneralisedPriorTests(char const* density_filename = nullptr);
  typedef DiscretisedDensity<3, float> target_type;
  void construct_input_data(shared_ptr<target_type>& density_sptr, shared_ptr<target_type>& kappa_sptr);

  //! Set methods that control which tests are run.
  void configure_prior_tests(bool gradient, bool Hessian_convexity, bool Hessian_numerical);

protected:
  char const* density_filename;
  shared_ptr<GeneralisedPrior<target_type>> objective_function_sptr;

  //! run the test
  /*! Note that this function is not specific to a particular prior */
  void run_tests_for_objective_function(const std::string& test_name,
                                        GeneralisedPrior<target_type>& objective_function,
                                        const shared_ptr<target_type>& target_sptr);

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
                                            float input_multiplication,
                                            float input_addition,
                                            float current_image_multiplication,
                                            float current_image_addition);

  //! Variables to control which tests are run, see the set methods
  //@{
  bool do_test_gradient = false;
  bool do_test_Hessian_convexity = false;
  bool do_test_Hessian_against_numerical = false;
  //@}
};

GeneralisedPriorTests::GeneralisedPriorTests(char const* const density_filename)
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
GeneralisedPriorTests::run_tests_for_objective_function(const std::string& test_name,
                                                        GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                                        const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  std::cerr << "----- test " << test_name << '\n';
  if (!check(objective_function.set_up(target_sptr) == Succeeded::yes, "set-up of objective function"))
    return;

  if (do_test_gradient)
    {
      std::cerr << "----- test " << test_name << "  --> Gradient\n";
      using value_type = target_type::full_value_type;
      const auto eps = static_cast<value_type>(1e-4F * target_sptr->find_max());
      test_gradient(test_name, objective_function, *target_sptr, eps);
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
      std::cerr << "----- test " << test_name << "  --> Hessian-vector product (accumulate_Hessian_times_input)\n";
      test_Hessian(test_name, objective_function, *target_sptr, 0.00001F);
    }
}

void
GeneralisedPriorTests::test_Hessian_convexity(const std::string& test_name,
                                              GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                              const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  if (!objective_function.is_convex())
    return;
  /// Construct configurations
  float beta_array[] = { 0.01, 1, 100 }; // Penalty strength should only affect scale
  // Modifications to the input image
  float input_multiplication_array[] = { -100, -1, 0.01, 1, 100 }; // Test negative, small and large values
  float input_addition_array[] = { -10, -1, -0.5, 0.0, 1, 10 };
  // Modifications to the current image (Hessian computation)
  float current_image_multiplication_array[] = { 0.01, 1, 100 };
  float current_image_addition_array[] = { 0.0, 0.5, 1, 10 }; // RDP has constraint that current_image is non-negative

  bool testOK = true;
  float initial_beta = objective_function.get_penalisation_factor();
  for (float beta : beta_array)
    for (float input_multiplication : input_multiplication_array)
      for (float input_addition : input_addition_array)
        for (float current_image_multiplication : current_image_multiplication_array)
          for (float current_image_addition : current_image_addition_array)
            {
              if (testOK) // only compute configuration if testOK from previous tests
                testOK = test_Hessian_convexity_configuration(test_name,
                                                              objective_function,
                                                              target_sptr,
                                                              beta,
                                                              input_multiplication,
                                                              input_addition,
                                                              current_image_multiplication,
                                                              current_image_addition);
            }
  /// Reset beta to original value
  objective_function.set_penalisation_factor(initial_beta);
}

bool
GeneralisedPriorTests::test_Hessian_convexity_configuration(
    const std::string& test_name,
    GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
    const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr,
    const float beta,
    const float input_multiplication,
    const float input_addition,
    const float current_image_multiplication,
    const float current_image_addition)
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
        ++input_iter;
        ++target_iter;
        ++current_image_iter;
      }
  }

  /// Compute H x
  objective_function.accumulate_Hessian_times_input(*output, *current_image, *input);

  /// Compute x \cdot (H x)
  const double my_sum = std::inner_product(input->begin_all(), input->end_all(), output->begin_all(), double(0));
  /// Compute x \cdot x
  const double my_norm2 = std::inner_product(input->begin_all(), input->end_all(), input->begin_all(), double(0));

  // test for a CONVEX function: 0 < my_sum, but we allow for some numerical error
  if (this->check_if_less(-my_norm2 * 2E-4, my_sum))
    {
      return true;
    }
  else
    {
      // print to console the FAILED configuration
      info("FAIL: Computation of x^T H x = " + std::to_string(my_sum) + " < 0 and is therefore NOT convex" + "\ntest_name="
           + test_name + "\nbeta=" + std::to_string(beta) + "\ninput_multiplication=" + std::to_string(input_multiplication)
           + "\ninput_addition=" + std::to_string(input_addition) + "\ncurrent_image_multiplication="
           + std::to_string(current_image_multiplication) + "\ncurrent_image_addition=" + std::to_string(current_image_addition)
           + "\n >input image max=" + std::to_string(input->find_max()) + "\n >input image min="
           + std::to_string(input->find_min()) + "\n >input image norm^2=" + std::to_string(my_norm2) + "\n >target image max="
           + std::to_string(target.find_max()) + "\n >target image min=" + std::to_string(target.find_min()));
      return false;
    }
}

void
GeneralisedPriorTests::test_Hessian_against_numerical(const std::string& test_name,
                                                      GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                                      const shared_ptr<GeneralisedPriorTests::target_type>& target_sptr)
{
  if (!objective_function.is_convex())
    return;

  /// Setup
  const float eps = 1e-4F * target_sptr->find_max();
  bool testOK = true;
  const int verbosity_default = Verbosity::get();

  // setup images
  shared_ptr<target_type> input_sptr(target_sptr->clone());
  auto& input(*input_sptr);
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
  for (int z = min_z; z <= max_z; z++)
    for (int y = min_y; y <= max_y; y++)
      for (int x = min_x; x <= max_x; x++)
        if (testOK)
          {
            perturbation_coords[1] = z;
            perturbation_coords[2] = y;
            perturbation_coords[3] = x;

            //  Compute H(x)_j (row of the Hessian at the jth voxel)
            objective_function.compute_Hessian(*Hessian_sptr, perturbation_coords, input);
            const double max_H = std::max(fabs(double(Hessian_sptr->find_min())), fabs(double(Hessian_sptr->find_max())));

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
            while (numerical_Hessian_iter != pert_grad_and_numerical_Hessian_sptr->end_all())
              {
                testOK
                    = testOK && this->check_if_less(std::abs(*Hessian_iter - *numerical_Hessian_iter), max_H * 0.005F, "Hessian");
                ++numerical_Hessian_iter;
                ++Hessian_iter;
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
GeneralisedPriorTests::construct_input_data(shared_ptr<target_type>& density_sptr, shared_ptr<target_type>& kappa_sptr)
{
  if (this->density_filename == nullptr)
    {
      // construct a small image with random voxel values between 0 and 1

      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      exam_info_sptr->imaging_modality = ImagingModality::PT;
      CartesianCoordinate3D<float> origin(0, 0, 0);
      CartesianCoordinate3D<float> voxel_size(2.F, 3.F, 3.F);

      density_sptr.reset(
          new VoxelsOnCartesianGrid<float>(exam_info_sptr, IndexRange<3>(make_coordinate(10, 9, 8)), origin, voxel_size));
      // fill with random numbers between 0 and 1
      typedef boost::mt19937 base_generator_type;
      // initialize by reproducible seed
      static base_generator_type generator(boost::uint32_t(42));
      static boost::uniform_01<base_generator_type> random01(generator);
      for (target_type::full_iterator iter = density_sptr->begin_all(); iter != density_sptr->end_all(); ++iter)
        *iter = static_cast<float>(random01());
    }
  else
    {
      // load image from file
      shared_ptr<target_type> aptr(read_from_file<target_type>(this->density_filename));
      density_sptr = aptr;
    }

  // create (unrealistic) kappa by filtering the original
  kappa_sptr = shared_ptr<target_type>(density_sptr->clone());
  SeparableGaussianImageFilter<float> filter;
  filter.set_fwhms(make_coordinate(25.F, 36.F, 27.F));
  filter.set_up(*kappa_sptr);
  filter.apply(*kappa_sptr);
}

/*!
 \brief tests for QuadraticPrior
 \ingroup recontest
 \ingroup priors
*/
class QuadraticPriorTests : public GeneralisedPriorTests
{
public:
  using GeneralisedPriorTests::GeneralisedPriorTests;
  void run_tests() override;
};

void
QuadraticPriorTests::run_tests()
{
  shared_ptr<target_type> density_sptr;
  shared_ptr<target_type> kappa_sptr;
  construct_input_data(density_sptr, kappa_sptr);

  std::cerr << "\n\nTests for QuadraticPrior\n";
  {
    QuadraticPrior<float> objective_function(false, 1.F);
    this->configure_prior_tests(true, true, true);
    this->run_tests_for_objective_function("Quadratic_no_kappa", objective_function, density_sptr);
    objective_function.set_kappa_sptr(kappa_sptr);
    this->run_tests_for_objective_function("Quadratic_with_kappa", objective_function, density_sptr);
  }
}

/*!
 \brief tests for RelativeDifferencePrior
 \ingroup recontest
 \ingroup priors
*/
template <class RDP>
class RelativeDifferencePriorTests : public GeneralisedPriorTests
{
public:
  using GeneralisedPriorTests::GeneralisedPriorTests;
  void run_specific_tests(const std::string& test_name,
                          RelativeDifferencePrior<float>& rdp,
                          const shared_ptr<target_type>& target_sptr);
  void run_tests() override;
};

template <class RDP>
void
RelativeDifferencePriorTests<RDP>::run_specific_tests(const std::string& test_name,
                                                      RelativeDifferencePrior<float>& rdp,
                                                      const shared_ptr<DiscretisedDensity<3, float>>& target_sptr)
{
  std::cerr << "----- test " << test_name << "  --> RDP gradient limit tests\n";
  shared_ptr<target_type> grad_sptr(target_sptr->get_empty_copy());
  const Array<3, float> weights = rdp.get_weights() * rdp.get_penalisation_factor();
  const bool do_kappa = rdp.get_kappa_sptr() != 0;
  // strictly speaking, we should be checking product of the kappas in a neighbourhood, but they usually very smoothly. In any
  // case, this will give an upper-bound
  const double kappa2_max = do_kappa ? square(rdp.get_kappa_sptr()->find_max()) : 1.;
  const auto weights_sum = weights.sum();

  if (rdp.get_epsilon() > 0)
    {
      // test Lipschitz condition on current image
      const double grad_Lipschitz = 4 * weights_sum * kappa2_max / rdp.get_epsilon();

      rdp.compute_gradient(*grad_sptr, *target_sptr);
      check_if_less(norm(grad_sptr->begin_all(), grad_sptr->end_all()),
                    grad_Lipschitz * norm(target_sptr->begin_all(), target_sptr->end_all()) * 1.001F,
                    "gradient Lipschitz with x = input_image, y = 0");
    }

  // do some checks on a "delta" image
  shared_ptr<target_type> delta_sptr(target_sptr->get_empty_copy());
  delta_sptr->fill(0.F);

  {
    // The derivative of the RDP_potential(x,0) limits to 1/(1+gamma). Therefore, the
    // gradient of the prior will limit to
    const auto grad_limit_no_kappa = weights_sum / (1 + rdp.get_gamma());

    const auto scale = rdp.get_epsilon() ? rdp.get_epsilon() : 1;
    auto idx = make_coordinate(1, 1, 1);
    double kappa_at_idx_2 = do_kappa ? square((*rdp.get_kappa_sptr())[idx]) : 1.;
    auto grad_limit = grad_limit_no_kappa * kappa_at_idx_2;
    (*delta_sptr)[idx] = 1E5F * scale;
    rdp.compute_gradient(*grad_sptr, *delta_sptr);
    check_if_less(std::abs((*grad_sptr)[idx] / grad_limit - 1), do_kappa ? 0.03 : 1e-4, "RDP gradient large limit");
    (*delta_sptr)[idx] = 1E20F * scale;
    rdp.compute_gradient(*grad_sptr, *delta_sptr);
    check_if_less(std::abs((*grad_sptr)[idx] / grad_limit - 1), do_kappa ? 0.03 : 1e-4, "RDP gradient very large limit");

    // check at boundary (fewer neighbours)
    idx = make_coordinate(0, 0, 0);
    (*delta_sptr)[idx] = 1E5F * scale;
    kappa_at_idx_2 = do_kappa ? square((*rdp.get_kappa_sptr())[idx]) : 1.;
    grad_limit = grad_limit_no_kappa * kappa_at_idx_2;
    rdp.compute_gradient(*grad_sptr, *delta_sptr);
    check_if_less((*grad_sptr)[idx] / grad_limit, 1., "RDP gradient large limit at boundary");
  }
}

template <class RDP>
void
RelativeDifferencePriorTests<RDP>::run_tests()
{
  shared_ptr<target_type> density_sptr;
  shared_ptr<target_type> kappa_sptr;
  construct_input_data(density_sptr, kappa_sptr);
  const std::string name(RDP::registered_name);
  std::cerr << "\n\nTests for " << name << " with epsilon = 0\n";
  {
    // gamma is default and epsilon is 0.0
    RDP objective_function(false, 1.F, 2.F, 0.F);
    this->configure_prior_tests(
        true, true, false); // RDP, with epsilon = 0.0, will fail the numerical Hessian test (it can become infinity)
    this->run_tests_for_objective_function(name + "_no_kappa_no_eps", objective_function, density_sptr);
    this->run_specific_tests(name + "_specific_no_kappa_no_eps", objective_function, density_sptr);
    objective_function.set_kappa_sptr(kappa_sptr);
    this->run_tests_for_objective_function(name + "_with_kappa_no_eps", objective_function, density_sptr);
    this->run_specific_tests(name + "_specific_with_kappa_no_eps", objective_function, density_sptr);
  }
  std::cerr << "\n\nTests for " << name << " with epsilon = 0.1\n";
  {
    // gamma is default and epsilon is "small"
    RDP objective_function(false, 1.F, 2.F, 0.1F);
    this->configure_prior_tests(true, true, true); // With a large enough epsilon the RDP Hessian numerical test will pass
    this->run_tests_for_objective_function(name + "_no_kappa_with_eps", objective_function, density_sptr);
    this->run_specific_tests(name + "_specific_no_kappa_with_eps", objective_function, density_sptr);
    objective_function.set_kappa_sptr(kappa_sptr);
    this->run_tests_for_objective_function(name + "_with_kappa_with_eps", objective_function, density_sptr);
    this->run_specific_tests(name + "_specific_with_kappa_with_eps", objective_function, density_sptr);
  }
}

/*!
 \brief tests for PLSPrior
 \ingroup recontest
 \ingroup priors
*/
class PLSPriorTests : public GeneralisedPriorTests
{
public:
  using GeneralisedPriorTests::GeneralisedPriorTests;
  void run_tests() override;
};

void
PLSPriorTests::run_tests()
{
  shared_ptr<target_type> density_sptr;
  shared_ptr<target_type> kappa_sptr;
  construct_input_data(density_sptr, kappa_sptr);

  std::cerr << "\n\nTests for PLSPrior\n";
  {
    PLSPrior<float> objective_function(false, 1.F);
    shared_ptr<DiscretisedDensity<3, float>> anatomical_image_sptr(density_sptr->get_empty_copy());
    anatomical_image_sptr->fill(1.F);
    objective_function.set_anatomical_image_sptr(anatomical_image_sptr);
    // Disabled PLS due to known issue
    this->configure_prior_tests(false, false, false);
    this->run_tests_for_objective_function("PLS_no_kappa_flat_anatomical", objective_function, density_sptr);
  }
}

/*!
 \brief tests for LogCoshPrior
 \ingroup recontest
 \ingroup priors
*/
class LogCoshPriorTests : public GeneralisedPriorTests
{
public:
  using GeneralisedPriorTests::GeneralisedPriorTests;
  void run_tests() override;
};

void
LogCoshPriorTests::run_tests()
{
  shared_ptr<target_type> density_sptr;
  shared_ptr<target_type> kappa_sptr;
  construct_input_data(density_sptr, kappa_sptr);

  std::cerr << "\n\nTests for Logcosh Prior\n";
  {
    // scalar is off
    LogcoshPrior<float> objective_function(false, 1.F, 1.F);
    this->configure_prior_tests(true, true, true);
    this->run_tests_for_objective_function("Logcosh_no_kappa", objective_function, density_sptr);
    objective_function.set_kappa_sptr(kappa_sptr);
    this->run_tests_for_objective_function("Logcosh_with_kappa", objective_function, density_sptr);
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  // option processing
  bool do_cuda_tests = true;
  while (argc > 1 && strncmp(argv[1], "--", 2) == 0)
    {
      if (strcmp(argv[1], "--help") == 0)
        {
          std::cerr << "Usage:\n"
                    << "    test_priors [--skip-cuda] [image_filename]\n";
          exit(EXIT_SUCCESS);
        }
      else if (strcmp(argv[1], "--skip-cuda") == 0)
        do_cuda_tests = false;
      else
        {
          std::cerr << "Unknown option: " << argv[1] << "\nUse --help for more information\n";
          exit(EXIT_FAILURE);
        }
      --argc;
      ++argv;
    }

  set_default_num_threads();

  bool everything_ok = true;

  {
    QuadraticPriorTests tests(argc > 1 ? argv[1] : nullptr);
    // tests.run_tests();
    everything_ok = everything_ok && tests.is_everything_ok();
  }
  {
    RelativeDifferencePriorTests<RelativeDifferencePrior<float>> tests(argc > 1 ? argv[1] : nullptr);
    tests.run_tests();
    everything_ok = everything_ok && tests.is_everything_ok();
  }
#ifdef STIR_WITH_CUDA
  if (do_cuda_tests)
    {
      RelativeDifferencePriorTests<CudaRelativeDifferencePrior<float>> tests(argc > 1 ? argv[1] : nullptr);
      tests.run_tests();
      everything_ok = everything_ok && tests.is_everything_ok();
    }
#endif
  {
    PLSPriorTests tests(argc > 1 ? argv[1] : nullptr);
    tests.run_tests();
    everything_ok = everything_ok && tests.is_everything_ok();
  }
  {
    LogCoshPriorTests tests(argc > 1 ? argv[1] : nullptr);
    tests.run_tests();
    everything_ok = everything_ok && tests.is_everything_ok();
  }

  if (!everything_ok)
    std::cerr << "Tests for at least 1 prior failed.\n";
  return everything_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
