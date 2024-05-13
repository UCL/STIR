/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2020, 2022-2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_test

  \brief Test skeleton for stir::GeneralisedObjectiveFunction and stir::GeneralisedPrior

  \author Kris Thielemans
  \author Reobert Twyman Skelly
*/

#include "stir/RunTests.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include <iostream>

START_NAMESPACE_STIR

/*!
  \ingroup recon_test
  \brief Test class for GeneralisedObjectiveFunction and GeneralisedPrior

  This contains some numerical tests to check gradient and Hessian calculations.

  Note that the gradient is computed numerically voxel by voxel which is obviously
  terribly slow. A solution (for the test) would be to compute it only in
  a subset of voxels or so. We'll leave this for later.

  Note that the test only works if the objective function is well-defined. For example,
  if certain projections are non-zero, while the model estimates them to be zero, the
  Poisson objective function is in theory infinite.
  ObjectiveFunction uses some thresholds to try to
  avoid overflow, but if there are too many of these bins, the total objective
  function will become infinite. The numerical gradient then becomes ill-defined
  (even in voxels that do not contribute to these bins).

*/
template <class ObjectiveFunctionT, class TargetT>
class ObjectiveFunctionTests : public RunTests
{
public:
  typedef ObjectiveFunctionT objective_function_type;
  typedef TargetT target_type;

  //! Test the gradient of the objective function by comparing to the numerical gradient via perturbation
  /*! Note: \a target is non-\c const, as the code will add/subtract eps, but the actual values
    are not modified after the test exits.
  */
  virtual Succeeded
  test_gradient(const std::string& test_name, ObjectiveFunctionT& objective_function, TargetT& target, const float eps);

  //! Test the accumulate_Hessian_times_input of the objective function by comparing to the numerical result via perturbation
  /*!
    This test checks that \f$ H dx \approx G(x+dx) - G(x) \f$.
    \f$dx\f$ is currently computed by as <code>eps*(target / target.find_max() + 0.5)</code>.
  */
  virtual Succeeded
  test_Hessian(const std::string& test_name, ObjectiveFunctionT& objective_function, const TargetT& target, const float eps);
  //! Test the Hessian of the objective function by testing the (\a mult_factor * x^T Hx > 0) condition
  virtual Succeeded test_Hessian_concavity(const std::string& test_name,
                                           ObjectiveFunctionT& objective_function,
                                           const TargetT& target,
                                           const float mult_factor = 1.F);
};

template <class ObjectiveFunctionT, class TargetT>
Succeeded
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_gradient(const std::string& test_name,
                                                                   ObjectiveFunctionT& objective_function,
                                                                   TargetT& target,
                                                                   const float eps)
{
  shared_ptr<TargetT> gradient_sptr(target.get_empty_copy());
  shared_ptr<TargetT> gradient_2_sptr(target.get_empty_copy());
  info("Computing gradient");
  objective_function.compute_gradient(*gradient_sptr, target);
  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), double(gradient_sptr->find_max())) / 1000);
  info("Computing objective function at target");
  const double value_at_target = objective_function.compute_value(target);
  auto target_iter = target.begin_all();
  auto gradient_iter = gradient_sptr->begin_all();
  auto gradient_2_iter = gradient_2_sptr->begin_all();
  bool testOK = true;
  info("Computing gradient of objective function by numerical differences (this will take a while)");
  while (target_iter != target.end_all())
    {
      *target_iter += eps;
      const double value_at_inc = objective_function.compute_value(target);
      *target_iter -= eps;
      const float gradient_at_iter = static_cast<float>((value_at_inc - value_at_target) / eps);
      *gradient_2_iter++ = gradient_at_iter;
      testOK = testOK && this->check_if_equal(gradient_at_iter, *gradient_iter, "gradient");
      ++target_iter;
      ++gradient_iter;
    }
  if (!testOK)
    {
      std::cerr << "Numerical gradient test failed with for " + test_name + "\n";
      std::cerr << "Writing diagnostic files " << test_name << "_target.hv, *gradient.hv, *numerical_gradient.hv\n";
      write_to_file(test_name + "_target.hv", target);
      write_to_file(test_name + "_gradient.hv", *gradient_sptr);
      write_to_file(test_name + "_numerical_gradient.hv", *gradient_2_sptr);
      return Succeeded::no;
    }
  else
    {
      return Succeeded::yes;
    }
}

template <class ObjectiveFunctionT, class TargetT>
Succeeded
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_Hessian(const std::string& test_name,
                                                                  ObjectiveFunctionT& objective_function,
                                                                  const TargetT& target,
                                                                  const float eps)
{
  /* test G(x+dx) = G(x) + H dx + small stuff */
  shared_ptr<TargetT> gradient_sptr(target.get_empty_copy());
  shared_ptr<TargetT> gradient_2_sptr(target.get_empty_copy());
  shared_ptr<TargetT> output(target.get_empty_copy());
  shared_ptr<TargetT> increment_sptr(target.clone());
  *increment_sptr *= eps / increment_sptr->find_max();
  *increment_sptr += eps / 2;
  shared_ptr<TargetT> target_plus_inc_sptr(target.clone());
  *target_plus_inc_sptr += *increment_sptr;

  info("Computing gradient");
  objective_function.compute_gradient(*gradient_sptr, target);
  objective_function.compute_gradient(*gradient_2_sptr, *target_plus_inc_sptr);
  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), double(gradient_sptr->find_max())) / 1E5);
  info("Computing Hessian * increment at target");
  objective_function.accumulate_Hessian_times_input(*output, target, *increment_sptr);
  auto output_iter = output->begin_all_const();
  auto gradient_iter = gradient_sptr->begin_all_const();
  auto gradient_2_iter = gradient_2_sptr->begin_all_const();
  bool testOK = true;
  info("Computing gradient of objective function by numerical differences (this will take a while)");
  while (output_iter != output->end_all())
    {
      testOK = testOK && this->check_if_equal(*gradient_2_iter - *gradient_iter, *output_iter, "Hessian*increment");
      ++output_iter;
      ++gradient_iter;
      ++gradient_2_iter;
    }
  if (!testOK)
    {
      std::cerr << "Numerical Hessian test failed with for " + test_name + "\n";
      std::cerr << "Writing diagnostic files " << test_name
                << "_target.hv, *gradient.hv, *increment, *numerical_gradient.hv, *Hessian_times_increment\n";
      write_to_file(test_name + "_target.hv", target);
      write_to_file(test_name + "_gradient.hv", *gradient_sptr);
      write_to_file(test_name + "_increment.hv", *increment_sptr);
      write_to_file(test_name + "_gradient_at_increment.hv", *gradient_2_sptr);
      write_to_file(test_name + "_Hessian_times_increment.hv", *output);
      return Succeeded::no;
    }
  else
    {
      return Succeeded::yes;
    }
}

template <class ObjectiveFunctionT, class TargetT>
Succeeded
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_Hessian_concavity(const std::string& test_name,
                                                                            ObjectiveFunctionT& objective_function,
                                                                            const TargetT& target,
                                                                            const float mult_factor)
{
  /// setup images
  shared_ptr<TargetT> output(target.get_empty_copy());

  /// Compute H x
  objective_function.accumulate_Hessian_times_input(*output, target, target);

  /// Compute dot(x,(H x))
  const float my_sum = std::inner_product(target.begin_all(), target.end_all(), output->begin_all(), 0.F) * mult_factor;

  // test for a CONCAVE function
  if (this->check_if_less(my_sum, 0))
    {
      //    info("PASS: Computation of x^T H x = " + std::to_string(my_sum) + " < 0" (Hessian) and is therefore concave);
      return Succeeded::yes;
    }
  else
    {
      // print to console the FAILED configuration
      std::cerr << "FAIL: " + test_name + ": Computation of x^T H x = " + std::to_string(my_sum)
                << " > 0 (Hessian) and is therefore NOT concave"
                << "\n >target image max=" << target.find_max() << "\n >target image min=" << target.find_min()
                << "\n >output image max=" << output->find_max() << "\n >output image min=" << output->find_min();
      std::cerr << "Writing diagnostic files to " << test_name + "_concavity_out.hv etc\n.";
      write_to_file(test_name + "_concavity_out.hv", *output);
      write_to_file(test_name + "_target.hv", target);
      return Succeeded::no;
    }
}

END_NAMESPACE_STIR
