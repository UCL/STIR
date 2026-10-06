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
  \brief implementation of the stir::OrderedSubsetsGradientEstimator class
  \author Manil Sabeur
*/

#include "stir/recon_buildblock/gradient_estimators/OrderedSubsetsGradientEstimator.h"
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

START_NAMESPACE_STIR

template <typename TargetT>
const char* const OrderedSubsetsGradientEstimator<TargetT>::registered_name = "Ordered Subsets";

template <typename TargetT>
OrderedSubsetsGradientEstimator<TargetT>::OrderedSubsetsGradientEstimator()
{
  set_defaults();
}

template <typename TargetT>
void
OrderedSubsetsGradientEstimator<TargetT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename TargetT>
void
OrderedSubsetsGradientEstimator<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Ordered Subsets Parameters");
  this->parser.add_stop_key("End Ordered Subsets Parameters");
}

template <typename TargetT>
Succeeded
OrderedSubsetsGradientEstimator<TargetT>::set_up(GeneralisedObjectiveFunction<TargetT>& objective_function,
                                                 const TargetT& /*target*/)
{
  this->objective_ptr = &objective_function;
  return Succeeded::yes;
}

template <typename TargetT>
void
OrderedSubsetsGradientEstimator<TargetT>::compute(
    TargetT& gradient, const TargetT& current_estimate, int /*update_num*/, int subset_num, int num_subsets)
{
  if (is_null_ptr(this->objective_ptr))
    error("OrderedSubsetsGradientEstimator: set_up() has to be called before compute()");
  info(format("Ordered Subsets: processing subset #: {}", subset_num));
  this->objective_ptr->compute_sub_gradient(gradient, current_estimate, subset_num);
  const float scale = static_cast<float>(num_subsets);
  std::transform(gradient.begin_all(), gradient.end_all(), gradient.begin_all(), _1 * scale);
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class OrderedSubsetsGradientEstimator<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
