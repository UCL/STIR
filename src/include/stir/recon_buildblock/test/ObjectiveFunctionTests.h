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
  \author Robert Twyman Skelly
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
  /*!
    If \a full_gradient=true, all elements in the gradient are tested (using single-element increments). This is slow.
    Otherwise, the test checks that \f$ G dx \approx F(x+dx) - F(x) \f$.  dx is computed via construct_increment().

    Note: \a target is non-\c const, as the code will add/subtract eps, but the actual values
    are not modified after the test exits.
  */
  virtual Succeeded test_gradient(const std::string& test_name,
                                  ObjectiveFunctionT& objective_function,
                                  TargetT& target,
                                  const float eps,
                                  const bool full_gradient = true);

  //! Test the accumulate_Hessian_times_input of the objective function by comparing to the numerical result via perturbation
  /*!
    This test checks that \f$ H dx \approx G(x+dx) - G(x) \f$. dx is computed via construct_increment().
  */
  virtual Succeeded
  test_Hessian(const std::string& test_name, ObjectiveFunctionT& objective_function, const TargetT& target, const float eps);
  //! Test the Hessian of the objective function by testing the (\a mult_factor * x^T Hx > 0) condition
  virtual Succeeded test_Hessian_concavity(const std::string& test_name,
                                           ObjectiveFunctionT& objective_function,
                                           const TargetT& target,
                                           const float mult_factor = 1.F);

protected:
  //! Construct small increment for target
  /*!
    Result is <code>eps*(target / target.find_max() + 0.5)</code>, i.e. it is always
    positive (if target is non-negative).
  */
  virtual shared_ptr<const TargetT> construct_increment(const TargetT& target, const float eps) const;
};

template <class ObjectiveFunctionT, class TargetT>
Succeeded
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_gradient(const std::string& test_name,
                                                                   ObjectiveFunctionT& objective_function,
                                                                   TargetT& target,
                                                                   const float eps,
                                                                   const bool full_gradient)
{
  shared_ptr<TargetT> gradient_sptr(target.get_empty_copy());
  shared_ptr<TargetT> gradient_2_sptr(target.get_empty_copy());
  info("Computing gradient");
  objective_function.compute_gradient(*gradient_sptr, target);
  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), double(gradient_sptr->find_max())) / 1000);
  info("Computing objective function at target");
  const double value_at_target = objective_function.compute_value(target);
  bool testOK = true;
  if (full_gradient)
    {
      info("Computing gradient of objective function by numerical differences (this will take a while)");
      auto target_iter = target.begin_all();
      auto gradient_iter = gradient_sptr->begin_all();
      auto gradient_2_iter = gradient_2_sptr->begin_all();
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
    }
  else
    {
      /* test f(x+dx) - f(x) ~ dx^t G(x) */
      shared_ptr<const TargetT> increment_sptr = this->construct_increment(target, eps);
      shared_ptr<TargetT> target_plus_inc_sptr(target.clone());
      *target_plus_inc_sptr += *increment_sptr;
      const double value_at_inc = objective_function.compute_value(*target_plus_inc_sptr);
      const double my_sum = std::inner_product(
          gradient_sptr->begin_all_const(), gradient_sptr->end_all_const(), increment_sptr->begin_all_const(), 0.);

      testOK = testOK && this->check_if_equal(value_at_inc - value_at_target, my_sum, "gradient");
    }

  if (!testOK)
    {
      std::cerr << "Numerical gradient test failed with for " + test_name + "\n";
      std::cerr << "Writing diagnostic files " << test_name
                << "_target.hv, *gradient.hv (and *numerical_gradient.hv if full gradient test is used)\n";
      write_to_file(test_name + "_target.hv", target);
      write_to_file(test_name + "_gradient.hv", *gradient_sptr);
      if (full_gradient)
        write_to_file(test_name + "_numerical_gradient.hv", *gradient_2_sptr);
      return Succeeded::no;
    }
  else
    {
      return Succeeded::yes;
    }
}

template <class ObjectiveFunctionT, class TargetT>
shared_ptr<const TargetT>
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::construct_increment(const TargetT& target, const float eps) const
{
  shared_ptr<TargetT> increment_sptr(target.clone());
  *increment_sptr *= eps / increment_sptr->find_max();
  *increment_sptr += eps / 2;
  return increment_sptr;
}

template <class ObjectiveFunctionT, class TargetT>
Succeeded
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_Hessian(const std::string& test_name,
                                                                  ObjectiveFunctionT& objective_function,
                                                                  const TargetT& target,
                                                                  const float eps)
{
  info("Comparing Hessian*dx with difference of gradients");

  /* test G(x+dx) = G(x) + H dx + small stuff */
  shared_ptr<TargetT> gradient_sptr(target.get_empty_copy());
  shared_ptr<TargetT> gradient_2_sptr(target.get_empty_copy());
  shared_ptr<TargetT> output(target.get_empty_copy());
  shared_ptr<const TargetT> increment_sptr = this->construct_increment(target, eps);
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

  // test for a CONCAVE function (after multiplying with mult_factor of course)
  if (this->check_if_less(my_sum, 0))
    {
      return Succeeded::yes;
    }
  else
    {
      // print to console the FAILED configuration
      std::cerr << "FAIL: " + test_name + ": Computation of x^T H x = " + std::to_string(my_sum)
                << " > 0 (Hessian) and is therefore NOT concave"
                << "\n >target image max=" << target.find_max() << "\n >target image min=" << target.find_min()
                << "\n >output image max=" << output->find_max() << "\n >output image min=" << output->find_min() << '\n';
      std::cerr << "Writing diagnostic files to " << test_name + "_concavity_out.hv, *target.hv\n";
      write_to_file(test_name + "_concavity_out.hv", *output);
      write_to_file(test_name + "_target.hv", target);
      return Succeeded::no;
    }
}

END_NAMESPACE_STIR
