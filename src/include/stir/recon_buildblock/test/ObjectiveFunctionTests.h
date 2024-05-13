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
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include "stir/IO/OutputFileFormat.h"
START_NAMESPACE_STIR

/*!
  \ingroup recon_test
  \brief Test class for GeneralisedObjectiveFunction and GeneralisedPrior

  This is a somewhat preliminary implementation of a test that compares the result
  of ObjectiveFunction::compute_gradient
  with a numerical gradient computed by using the
  ObjectiveFunction::compute_objective_function() function.

  The trouble with this is that compute the gradient voxel by voxel is obviously
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
  virtual void
  test_gradient(const std::string& test_name, ObjectiveFunctionT& objective_function, TargetT& target, const float eps);

  //! Test the Hessian of the objective function by testing the (\a mult_factor * x^T Hx > 0) condition
  virtual void test_Hessian_concavity(const std::string& test_name,
                                      ObjectiveFunctionT& objective_function,
                                      TargetT& target,
                                      const float mult_factor = 1.F);
};

template <class ObjectiveFunctionT, class TargetT>
void
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
      info("Writing diagnostic files target.hv, gradient.hv, numerical_gradient.hv");
      write_to_file("target.hv", target);
      write_to_file("gradient.hv", *gradient_sptr);
      write_to_file("numerical_gradient.hv", *gradient_2_sptr);
    }
}

template <class ObjectiveFunctionT, class TargetT>
void
ObjectiveFunctionTests<ObjectiveFunctionT, TargetT>::test_Hessian_concavity(const std::string& test_name,
                                                                            ObjectiveFunctionT& objective_function,
                                                                            TargetT& target,
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
    }
  else
    {
      // print to console the FAILED configuration
      info("FAIL: " + test_name + ": Computation of x^T H x = " + std::to_string(my_sum)
           + " > 0 (Hessian) and is therefore NOT concave" + "\n >target image max=" + std::to_string(target.find_max())
           + "\n >target image min=" + std::to_string(target.find_min()));
    }
}

END_NAMESPACE_STIR
