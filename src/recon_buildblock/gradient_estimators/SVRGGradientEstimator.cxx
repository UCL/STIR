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
  \brief implementation of the stir::SVRGGradientEstimator class

  \author Manil Sabeur

  Native port of the SVRG estimator of the PETRIC2 "psv" contribution by Hok Shing Wong
  (Univ. of Bath), Margaret Duff (STFC), Matthias Ehrhardt (Univ. of Bath) and Georg Schramm
  (KU Leuven).
*/

#include "stir/recon_buildblock/gradient_estimators/SVRGGradientEstimator.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/format.h"

#include <algorithm>
#include "boost/lambda/lambda.hpp"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const SVRGGradientEstimator<TargetT>::registered_name = "SVRG";

template <typename TargetT>
SVRGGradientEstimator<TargetT>::SVRGGradientEstimator()
{
  set_defaults();
}

template <typename TargetT>
void
SVRGGradientEstimator<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->complete_gradient_epoch_interval = 2; // as in the PETRIC2 psv submission
}

template <typename TargetT>
void
SVRGGradientEstimator<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("SVRG Parameters");
  this->parser.add_key("complete gradient epoch interval", &this->complete_gradient_epoch_interval);
  this->parser.add_stop_key("End SVRG Parameters");
}

template <typename TargetT>
Succeeded
SVRGGradientEstimator<TargetT>::set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target)
{
  this->objective_ptr = &objective_function;
  const int num_subsets = objective_function.get_num_subsets();
  this->subset_gradients.resize(num_subsets);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    this->subset_gradients[subset_num].reset(target.get_empty_copy());
  this->summed_subset_gradients_sptr.reset(target.get_empty_copy());
  return Succeeded::yes;
}

template <typename TargetT>
void
SVRGGradientEstimator<TargetT>::compute(
    TargetT& gradient, const TargetT& current_estimate, int update_num, int subset_num, int num_subsets)
{
  if (is_null_ptr(this->objective_ptr))
    error("SVRGGradientEstimator: set_up() has to be called before compute()");
  if (subset_num < 0 || subset_num >= static_cast<int>(this->subset_gradients.size())
      || num_subsets != static_cast<int>(this->subset_gradients.size()))
    error(format("SVRGGradientEstimator: compute() called for subset {} of {}, but set_up() prepared {} subset "
                 "gradients",
                 subset_num,
                 num_subsets,
                 this->subset_gradients.size()));
  const int epoch = update_num / num_subsets;
  const bool at_epoch_start = (update_num % num_subsets == 0);
  const bool at_snapshot = at_epoch_start && (epoch % this->complete_gradient_epoch_interval == 0);

  if (at_snapshot)
    {
      // full SVRG snapshot: recompute all subset gradients and their sum, step with the full gradient
      info("SVRG: recomputing all subset gradients (snapshot)");
      std::fill(this->summed_subset_gradients_sptr->begin_all(), this->summed_subset_gradients_sptr->end_all(), 0.F);
      for (int s = 0; s < num_subsets; ++s)
        {
          this->objective_ptr->compute_sub_gradient(*this->subset_gradients[s], current_estimate, s);
          std::transform(this->summed_subset_gradients_sptr->begin_all(),
                         this->summed_subset_gradients_sptr->end_all(),
                         this->subset_gradients[s]->begin_all(),
                         this->summed_subset_gradients_sptr->begin_all(),
                         _1 + _2);
        }
      gradient = *this->summed_subset_gradients_sptr;
    }
  else
    {
      // g_tilde = num_subsets * (grad_i(x) - grad_i(x_snapshot)) + sum_j grad_j(x_snapshot)
      info(format("SVRG: processing subset #: {}", subset_num));
      this->objective_ptr->compute_sub_gradient(gradient, current_estimate, subset_num);
      std::transform(gradient.begin_all(),
                     gradient.end_all(),
                     this->subset_gradients[subset_num]->begin_all(),
                     gradient.begin_all(),
                     _1 - _2);
      std::transform(gradient.begin_all(),
                     gradient.end_all(),
                     this->summed_subset_gradients_sptr->begin_all(),
                     gradient.begin_all(),
                     _1 * num_subsets + _2);
    }
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class SVRGGradientEstimator<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
