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
  \brief implementation of the stir::SAGAGradientEstimator class
  \author Manil Sabeur
*/

#include "stir/recon_buildblock/gradient_estimators/SAGAGradientEstimator.h"
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
const char* const SAGAGradientEstimator<TargetT>::registered_name = "SAGA";

template <typename TargetT>
SAGAGradientEstimator<TargetT>::SAGAGradientEstimator()
{
  set_defaults();
}

template <typename TargetT>
void
SAGAGradientEstimator<TargetT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename TargetT>
void
SAGAGradientEstimator<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("SAGA Parameters");
  this->parser.add_stop_key("End SAGA Parameters");
}

template <typename TargetT>
Succeeded
SAGAGradientEstimator<TargetT>::set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target)
{
  this->objective_ptr = &objective_function;
  const int num_subsets = objective_function.get_num_subsets();
  this->table.resize(num_subsets);
  for (int s = 0; s < num_subsets; ++s)
    this->table[s].reset(target.get_empty_copy());
  this->summed_table_sptr.reset(target.get_empty_copy());
  this->scratch_sptr.reset(target.get_empty_copy());
  this->delta_sptr.reset(target.get_empty_copy());
  this->initialised = false;
  return Succeeded::yes;
}

template <typename TargetT>
void
SAGAGradientEstimator<TargetT>::compute(
    TargetT& gradient, const TargetT& current_estimate, int /*update_num*/, int subset_num, int num_subsets)
{
  if (is_null_ptr(this->objective_ptr))
    error("SAGAGradientEstimator: set_up() has to be called before compute()");
  if (subset_num < 0 || subset_num >= static_cast<int>(this->table.size()) || num_subsets != static_cast<int>(this->table.size()))
    error(format("SAGAGradientEstimator: compute() called for subset {} of {}, but set_up() prepared a table of {}",
                 subset_num,
                 num_subsets,
                 this->table.size()));
  if (!this->initialised)
    {
      // first update: initialise the table with a full pass; step with the full gradient
      info("SAGA: initialising the gradient table (full pass)");
      std::fill(this->summed_table_sptr->begin_all(), this->summed_table_sptr->end_all(), 0.F);
      for (int s = 0; s < num_subsets; ++s)
        {
          this->objective_ptr->compute_sub_gradient(*this->table[s], current_estimate, s);
          std::transform(this->summed_table_sptr->begin_all(),
                         this->summed_table_sptr->end_all(),
                         this->table[s]->begin_all(),
                         this->summed_table_sptr->begin_all(),
                         _1 + _2);
        }
      gradient = *this->summed_table_sptr;
      this->initialised = true;
      return;
    }

  info(format("SAGA: processing subset #: {}", subset_num));
  // new subset gradient
  this->objective_ptr->compute_sub_gradient(*this->scratch_sptr, current_estimate, subset_num);
  // delta = new - stored table entry
  std::transform(this->scratch_sptr->begin_all(),
                 this->scratch_sptr->end_all(),
                 this->table[subset_num]->begin_all(),
                 this->delta_sptr->begin_all(),
                 _1 - _2);
  // g_tilde = num_subsets * delta + sum, using the sum before this update
  std::transform(this->delta_sptr->begin_all(),
                 this->delta_sptr->end_all(),
                 this->summed_table_sptr->begin_all(),
                 gradient.begin_all(),
                 _1 * num_subsets + _2);
  // then update the running sum: sum += delta
  std::transform(this->summed_table_sptr->begin_all(),
                 this->summed_table_sptr->end_all(),
                 this->delta_sptr->begin_all(),
                 this->summed_table_sptr->begin_all(),
                 _1 + _2);
  // table entry <- new subset gradient
  *this->table[subset_num] = *this->scratch_sptr;
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class SAGAGradientEstimator<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
