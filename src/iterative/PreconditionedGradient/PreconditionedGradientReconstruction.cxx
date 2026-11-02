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
  \ingroup PreconditionedGradient
  \ingroup reconstructors
  \brief implementation of the stir::PreconditionedGradientReconstruction class
  \author Manil Sabeur
*/

#include "stir/PreconditionedGradient/PreconditionedGradientReconstruction.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

#include <memory>
#include <algorithm>
#include <sstream>
#include "boost/lambda/lambda.hpp"
#include "stir/unique_ptr.h"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const PreconditionedGradientReconstruction<TargetT>::registered_name = "Preconditioned Gradient";

template <class TargetT>
PreconditionedGradientReconstruction<TargetT>::PreconditionedGradientReconstruction()
{
  set_defaults();
}

template <class TargetT>
PreconditionedGradientReconstruction<TargetT>::PreconditionedGradientReconstruction(const std::string& parameter_filename)
{
  this->initialise(parameter_filename);
  info(this->parameter_info());
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_defaults()
{
  base_type::set_defaults();
  // defaults of the engine; a preset (e.g. PSV) overrides these.
  this->initial_step_size = 1.0;
  this->step_size_decay = 0.0;
  this->step_decay_per_epoch = false;
  this->pre_filter_fwhm_mm = 0.0;
  this->precond_update_epochs = std::vector<int>();
  this->precond_update_subiterations = std::vector<int>();
  this->precond_update_interval = 0;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PreconditionedGradientParameters");
  this->parser.add_stop_key("End");

  this->parser.add_key("initial step size", &this->initial_step_size);
  this->parser.add_key("step size decay", &this->step_size_decay);
  this->parser.add_key("step size decay per epoch", &this->step_decay_per_epoch);
  this->parser.add_key("initial image filter fwhm (mm)", &this->pre_filter_fwhm_mm);
  this->parser.add_key("precond update epochs", &this->precond_update_epochs);
  this->parser.add_key("precond update subiterations", &this->precond_update_subiterations);
  this->parser.add_key("precond update interval", &this->precond_update_interval);

  this->parser.add_parsing_key("preconditioner type", &this->preconditioner_sptr);
  this->parser.add_parsing_key("gradient estimator type", &this->gradient_estimator_sptr);
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_gradient_estimator_sptr(
    const shared_ptr<GeneralisedGradientEstimator<TargetT>>& arg)
{
  this->_already_set_up = false;
  this->gradient_estimator_sptr = arg;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_preconditioner_sptr(const shared_ptr<GeneralisedPreconditioner<TargetT>>& arg)
{
  this->_already_set_up = false;
  this->preconditioner_sptr = arg;
}

template <class TargetT>
const GeneralisedGradientEstimator<TargetT>&
PreconditionedGradientReconstruction<TargetT>::get_gradient_estimator() const
{
  if (is_null_ptr(this->gradient_estimator_sptr))
    error("PreconditionedGradientReconstruction: no gradient estimator has been set");
  return *this->gradient_estimator_sptr;
}

template <class TargetT>
const GeneralisedPreconditioner<TargetT>&
PreconditionedGradientReconstruction<TargetT>::get_preconditioner() const
{
  if (is_null_ptr(this->preconditioner_sptr))
    error("PreconditionedGradientReconstruction: no preconditioner has been set");
  return *this->preconditioner_sptr;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_initial_step_size(double arg)
{
  this->_already_set_up = false;
  this->initial_step_size = arg;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_step_size_decay(double arg)
{
  this->_already_set_up = false;
  this->step_size_decay = arg;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_step_size_decay_per_epoch(bool arg)
{
  this->_already_set_up = false;
  this->step_decay_per_epoch = arg;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_precond_update_interval(int arg)
{
  this->_already_set_up = false;
  this->precond_update_interval = arg;
}

template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::set_initial_image_filter_fwhm_mm(double arg)
{
  this->_already_set_up = false;
  this->pre_filter_fwhm_mm = arg;
}

template <class TargetT>
std::string
PreconditionedGradientReconstruction<TargetT>::method_info() const
{
  std::ostringstream s;
  if (!this->objective_function_sptr->prior_is_zero())
    s << "MAP-";
  if (this->num_subsets > 1)
    s << "OS-";
  s << "preconditioned gradient";
  // name the two axes: the log has to say which composition actually ran. Guarded because
  // method_info() can be called before set_up() has checked that both axes are present.
  if (!is_null_ptr(this->gradient_estimator_sptr) && !is_null_ptr(this->preconditioner_sptr))
    s << " (" << this->gradient_estimator_sptr->get_registered_name() << " + " << this->preconditioner_sptr->get_registered_name()
      << ")";
  return s.str();
}

template <class TargetT>
Succeeded
PreconditionedGradientReconstruction<TargetT>::set_up(shared_ptr<TargetT> const& target_image_ptr)
{
  if (base_type::set_up(target_image_ptr) == Succeeded::no)
    return Succeeded::no;

  // both axes have to be chosen explicitly: there is no default. (Presets such as PSV build their
  // own axes and call set_up_components() directly.)
  if (is_null_ptr(this->preconditioner_sptr))
    {
      warning("Preconditioned Gradient: no preconditioner selected. Set 'preconditioner type' in the .par "
              "(e.g. EM, Hessian Diagonal, Identity, SPS).\n");
      return Succeeded::no;
    }
  if (is_null_ptr(this->gradient_estimator_sptr))
    {
      warning("Preconditioned Gradient: no gradient estimator selected. Set 'gradient estimator type' in "
              "the .par (e.g. Full Gradient, Ordered Subsets, SVRG, SAGA).\n");
      return Succeeded::no;
    }

  // scale-compatibility contract: a subset-scaled preconditioner (P built from the current subset's
  // sensitivity) is only exact against the per-subset output of 'Ordered Subsets'. Refuse the
  // composition here, with a diagnosis, rather than silently mis-scaling the steps.
  if (this->preconditioner_sptr->is_subset_scaled() && !this->gradient_estimator_sptr->is_subset_gradient())
    {
      warning("Preconditioned Gradient: this preconditioner is subset-scaled (e.g. EM with 'use subset "
              "sensitivity') and requires the 'Ordered Subsets' estimator. Full-gradient estimators "
              "(Full Gradient, SVRG, SAGA) need a full-scale preconditioner (EM, Hessian Diagonal, SPS).\n");
      return Succeeded::no;
    }

  return this->set_up_components(target_image_ptr);
}

template <class TargetT>
Succeeded
PreconditionedGradientReconstruction<TargetT>::set_up_components(shared_ptr<TargetT> const& target_image_ptr)
{
  // set up the (already chosen) preconditioner and estimator
  if (this->preconditioner_sptr->set_up(*this->objective_function_sptr, *target_image_ptr) == Succeeded::no)
    {
      warning("Preconditioned Gradient: could not set up the preconditioner\n");
      return Succeeded::no;
    }
  if (this->gradient_estimator_sptr->set_up(*this->objective_function_sptr, *target_image_ptr) == Succeeded::no)
    {
      warning("Preconditioned Gradient: could not set up the gradient estimator\n");
      return Succeeded::no;
    }

  // pre-filter the initial image with an isotropic Gaussian (optional)
  if (this->pre_filter_fwhm_mm > 0)
    {
      SeparableGaussianImageFilter<float> pre_filter;
      BasicCoordinate<3, float> fwhms;
      fwhms[1] = fwhms[2] = fwhms[3] = static_cast<float>(this->pre_filter_fwhm_mm);
      pre_filter.set_fwhms(fwhms);
      if (pre_filter.set_up(*target_image_ptr) == Succeeded::no)
        {
          warning("Preconditioned Gradient: could not set up the initial image filter\n");
          return Succeeded::no;
        }
      pre_filter.apply(*target_image_ptr);
    }

  this->precond_sptr.reset(target_image_ptr->get_empty_copy()); // the P image, filled on first compute
  this->precond_computed = false;                               // force computation before the first update
  return Succeeded::yes;
}

/*! \brief preconditioned gradient additive update at every subiteration
  \warning This modifies the estimator's stored state. So you <strong>have to</strong> call set_up()
  before running a new reconstruction.
  */
template <class TargetT>
void
PreconditionedGradientReconstruction<TargetT>::update_estimate(TargetT& current_image_estimate)
{
  this->check(current_image_estimate);

  const int num_subsets = this->get_num_subsets();
  const int update_num = this->get_subiteration_num() - this->get_start_subiteration_num();
  const int epoch = update_num / num_subsets;
  const bool at_epoch_start = (update_num % num_subsets == 0);
  const int subset_num = this->get_subset_num();

  // preconditioner schedule: before the very first update, at the start of the listed epochs,
  // and at the listed update indices
  const bool at_precond_epoch = at_epoch_start
                                && std::find(this->precond_update_epochs.begin(), this->precond_update_epochs.end(), epoch)
                                       != this->precond_update_epochs.end();
  const bool at_precond_update
      = std::find(this->precond_update_subiterations.begin(), this->precond_update_subiterations.end(), update_num)
        != this->precond_update_subiterations.end();
  const bool at_precond_interval = this->precond_update_interval > 0 && (update_num % this->precond_update_interval == 0);
  if (!this->precond_computed || at_precond_epoch || at_precond_update || at_precond_interval)
    {
      this->preconditioner_sptr->compute(*this->precond_sptr, current_image_estimate, subset_num, num_subsets);
      this->precond_computed = true;
    }

  // gradient: delegated to the estimator, which owns any snapshot state and schedule
  unique_ptr<TargetT> gradient_ptr(current_image_estimate.get_empty_copy());
  this->gradient_estimator_sptr->compute(*gradient_ptr, current_image_estimate, update_num, subset_num, num_subsets);

  // decreasing step, preconditioned ascent.
  // \warning Two different epoch conventions meet here, on purpose. The preconditioner schedule
  // above counts updates *since the start of this run* (update_num), while the per-epoch decay
  // uses the *absolute* subiteration number, exactly as OSSPSReconstruction does
  // (relaxation_parameter / (1 + gamma * (subiteration_num / num_subsets))), so that the composed
  // engine reproduces OSSPS. On a restart ('start subiteration number' > 1) the two therefore
  // disagree: the schedule starts over while the step size continues to decay.
  const int decay_index = this->step_decay_per_epoch ? (this->get_subiteration_num() / num_subsets) : update_num;
  const float step_size = static_cast<float>(this->initial_step_size / (1. + this->step_size_decay * decay_index));
  info(format("step size = {}", step_size));
  std::transform(gradient_ptr->begin_all(),
                 gradient_ptr->end_all(),
                 this->precond_sptr->begin_all(),
                 gradient_ptr->begin_all(),
                 _1 * _2 * step_size);
  current_image_estimate += *gradient_ptr;

  // enforce the non-negativity constraint
  std::for_each(current_image_estimate.begin_all(), current_image_estimate.end_all(), [](float& a) {
    if (a < 0.F)
      a = 0.F;
  });
}

END_NAMESPACE_STIR

///////// instantiations
#include "stir/DiscretisedDensity.h"
START_NAMESPACE_STIR
template class PreconditionedGradientReconstruction<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
