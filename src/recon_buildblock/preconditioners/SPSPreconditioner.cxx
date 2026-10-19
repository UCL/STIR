//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief implementation of the stir::SPSPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/SPSPreconditioner.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/format.h"
#include "stir/unique_ptr.h"

#include <algorithm>
#include "boost/lambda/lambda.hpp"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const SPSPreconditioner<TargetT>::registered_name = "SPS";

template <typename TargetT>
SPSPreconditioner<TargetT>::SPSPreconditioner()
{
  set_defaults();
}

template <typename TargetT>
void
SPSPreconditioner<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->denominator_floor = 1e-6;
  this->surrogate_prior_ptr = nullptr;
}

template <typename TargetT>
void
SPSPreconditioner<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("SPS Preconditioner Parameters");
  this->parser.add_key("denominator floor", &this->denominator_floor);
  this->parser.add_stop_key("End SPS Preconditioner Parameters");
}

template <typename TargetT>
Succeeded
SPSPreconditioner<TargetT>::set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target)
{
  // capability contract: a prior, if present, must provide a parabolic surrogate.
  this->surrogate_prior_ptr = nullptr;
  if (!is_null_ptr(objective_function.get_prior_ptr()))
    {
      this->surrogate_prior_ptr = dynamic_cast<PriorWithParabolicSurrogate<TargetT>*>(objective_function.get_prior_ptr());
      if (is_null_ptr(this->surrogate_prior_ptr))
        {
          warning(format("SPS preconditioner: the prior '{}' does not provide a parabolic surrogate curvature. "
                         "Compatible priors (derived from PriorWithParabolicSurrogate): Quadratic, Logcosh, or no "
                         "prior. No implementation of the Relative Difference Prior provides one; the Gibbs "
                         "priors provide the Hessian diagonal instead (see the 'Hessian Diagonal' preconditioner).",
                         objective_function.get_prior_ptr()->get_registered_name()));
          return Succeeded::no;
        }
    }

  // precompute the (sign-flipped) row-sum of the approximate Hessian of the log-likelihood.
  this->precomputed_denominator_sptr.reset(target.get_empty_copy());
  std::fill(this->precomputed_denominator_sptr->begin_all(), this->precomputed_denominator_sptr->end_all(), 0.F);
  unique_ptr<TargetT> ones_ptr(target.clone());
  std::fill(ones_ptr->begin_all(), ones_ptr->end_all(), 1.F);
  if (objective_function.add_multiplication_with_approximate_Hessian_without_penalty(*this->precomputed_denominator_sptr,
                                                                                     *ones_ptr)
      == Succeeded::no)
    {
      warning("SPS preconditioner: could not compute the approximate Hessian of the likelihood\n");
      return Succeeded::no;
    }
  // the objective is concave (maximisation) -> the approximate Hessian is non-positive; flip to get a
  // non-negative denominator.
  std::for_each(
      this->precomputed_denominator_sptr->begin_all(), this->precomputed_denominator_sptr->end_all(), [](float& a) { a = -a; });

  return Succeeded::yes;
}

template <typename TargetT>
void
SPSPreconditioner<TargetT>::compute(TargetT& precond, const TargetT& current_estimate)
{
  info("recomputing preconditioner (SPS)");
  // denominator = 2 * parabolic_surrogate(x) + precomputed_likelihood_curvature
  unique_ptr<TargetT> denom_ptr(this->precomputed_denominator_sptr->clone());
  if (!is_null_ptr(this->surrogate_prior_ptr))
    {
      unique_ptr<TargetT> surrogate_ptr(current_estimate.get_empty_copy());
      this->surrogate_prior_ptr->parabolic_surrogate_curvature(*surrogate_ptr, current_estimate);
      std::transform(
          surrogate_ptr->begin_all(), surrogate_ptr->end_all(), denom_ptr->begin_all(), denom_ptr->begin_all(), _1 * 2 + _2);
    }
  // avoid division by (near-)zero, as native OSSPS does
  threshold_min_to_small_positive_value(
      denom_ptr->begin_all(), denom_ptr->end_all(), static_cast<float>(this->denominator_floor));
  // P = 1 / denominator
  std::transform(denom_ptr->begin_all(), denom_ptr->end_all(), precond.begin_all(), 1.F / _1);
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class SPSPreconditioner<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
