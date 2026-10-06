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
  \brief implementation of the stir::FullGradientEstimator class
  \author Manil Sabeur
*/

#include "stir/recon_buildblock/gradient_estimators/FullGradientEstimator.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"

#include <algorithm>
#include "boost/lambda/lambda.hpp"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const FullGradientEstimator<TargetT>::registered_name = "Full Gradient";

template <typename TargetT>
FullGradientEstimator<TargetT>::FullGradientEstimator()
{
  set_defaults();
}

template <typename TargetT>
void
FullGradientEstimator<TargetT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename TargetT>
void
FullGradientEstimator<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Full Gradient Parameters");
  this->parser.add_stop_key("End Full Gradient Parameters");
}

template <typename TargetT>
Succeeded
FullGradientEstimator<TargetT>::set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target)
{
  this->objective_ptr = &objective_function;
  this->scratch_sptr.reset(target.get_empty_copy());
  return Succeeded::yes;
}

template <typename TargetT>
void
FullGradientEstimator<TargetT>::compute(
    TargetT& gradient, const TargetT& current_estimate, int /*update_num*/, int /*subset_num*/, int num_subsets)
{
  if (is_null_ptr(this->objective_ptr))
    error("FullGradientEstimator: set_up() has to be called before compute()");
  info("Full Gradient: computing the full gradient (all subsets)");
  std::fill(gradient.begin_all(), gradient.end_all(), 0.F);
  for (int s = 0; s < num_subsets; ++s)
    {
      this->objective_ptr->compute_sub_gradient(*this->scratch_sptr, current_estimate, s);
      std::transform(gradient.begin_all(), gradient.end_all(), this->scratch_sptr->begin_all(), gradient.begin_all(), _1 + _2);
    }
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class FullGradientEstimator<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
