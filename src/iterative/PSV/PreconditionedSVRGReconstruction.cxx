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
\ingroup PSV
\ingroup reconstructors
\brief  implementation of the stir::PreconditionedSVRGReconstruction class

\author Manil Sabeur

  Native C++ port of the PETRIC2 "psv" contribution by Hok Shing Wong (Univ. of Bath),
  Margaret Duff (STFC), Matthias Ehrhardt (Univ. of Bath) and Georg Schramm (KU Leuven).
  See the class header for the full attribution and references.

  A locked preset of PreconditionedGradientReconstruction: the estimator is SVRG and the
  preconditioner is Hessian-diagonal, configured by the psv hyper-parameters. The estimator and
  preconditioner \c type keys are not exposed (use "Preconditioned Gradient" for that).
*/

#include "stir/PSV/PreconditionedSVRGReconstruction.h"
#include "stir/recon_buildblock/preconditioners/HessianDiagonalPreconditioner.h"
#include "stir/recon_buildblock/gradient_estimators/SVRGGradientEstimator.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/format.h"

#include <vector>

START_NAMESPACE_STIR

template <typename TargetT>
const char* const PreconditionedSVRGReconstruction<TargetT>::registered_name = "PSV";

template <class TargetT>
PreconditionedSVRGReconstruction<TargetT>::PreconditionedSVRGReconstruction()
{
  set_defaults();
}

template <class TargetT>
PreconditionedSVRGReconstruction<TargetT>::PreconditionedSVRGReconstruction(const std::string& parameter_filename)
{
  this->initialise(parameter_filename);
  info(this->parameter_info());
}

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::set_defaults()
{
  base_type::set_defaults();

  // the defaults are those of the original PETRIC2 "psv" submission; they can be overridden in the
  // parameter file. The step / filter / schedule members belong to the engine base.
  this->initial_step_size = 3.5;
  this->step_size_decay = 0.01;
  this->pre_filter_fwhm_mm = 6.0;
  this->precond_update_epochs = std::vector<int>(1, 1);
  this->precond_update_subiterations = std::vector<int>(1, 4);

  this->precond_hessian_factor = 2.0;
  this->precond_filter_fwhm_mm = 5.0;
  this->precond_delta_rel = 0.0;
  this->complete_gradient_epoch_interval = 2;
  this->adjoint_ones_floor = 1e-6;
  this->fov_mask_filename = "";
  this->fov_threshold = 1e-3;
}

template <class TargetT>
void
PreconditionedSVRGReconstruction<TargetT>::initialise_keymap()
{
  // locked preset: skip the engine's keymap (which would expose `preconditioner type` /
  // `gradient estimator type`) and go straight to IterativeReconstruction, then add only the
  // psv keys.
  IterativeReconstruction<TargetT>::initialise_keymap();
  this->parser.add_start_key("PSVParameters");
  this->parser.add_stop_key("End");

  // step / filter / schedule (engine members, re-exposed under the psv names)
  this->parser.add_key("initial step size", &this->initial_step_size);
  this->parser.add_key("step size decay", &this->step_size_decay);
  this->parser.add_key("initial image filter fwhm (mm)", &this->pre_filter_fwhm_mm);
  this->parser.add_key("precond update epochs", &this->precond_update_epochs);
  this->parser.add_key("precond update subiterations", &this->precond_update_subiterations);

  // psv hyper-parameters that configure the locked Hessian-diagonal + SVRG axes
  this->parser.add_key("precond hessian factor", &this->precond_hessian_factor);
  this->parser.add_key("precond filter fwhm (mm)", &this->precond_filter_fwhm_mm);
  this->parser.add_key("precond delta rel", &this->precond_delta_rel);
  this->parser.add_key("complete gradient epoch interval", &this->complete_gradient_epoch_interval);
  this->parser.add_key("adjoint ones floor", &this->adjoint_ones_floor);
  this->parser.add_key("FOV mask filename", &this->fov_mask_filename);
  this->parser.add_key("FOV threshold", &this->fov_threshold);
}

template <class TargetT>
bool
PreconditionedSVRGReconstruction<TargetT>::post_processing()
{
  if (base_type::post_processing())
    return true;

  if (this->initial_step_size <= 0)
    {
      warning(format("PSV: initial step size should be strictly positive but is {}", this->initial_step_size));
      return true;
    }
  if (this->step_size_decay < 0)
    {
      warning(format("PSV: step size decay should be non-negative but is {}", this->step_size_decay));
      return true;
    }
  if (this->precond_hessian_factor < 0)
    {
      warning(format("PSV: precond hessian factor should be non-negative but is {}", this->precond_hessian_factor));
      return true;
    }
  if (this->complete_gradient_epoch_interval < 1)
    {
      warning(
          format("PSV: complete gradient epoch interval should be at least 1 but is {}", this->complete_gradient_epoch_interval));
      return true;
    }
  return false;
}

template <class TargetT>
Succeeded
PreconditionedSVRGReconstruction<TargetT>::set_up(shared_ptr<TargetT> const& target_image_ptr)
{
  // set up the objective here: the engine's set_up() expects both axes to come from its own keys
  if (IterativeReconstruction<TargetT>::set_up(target_image_ptr) == Succeeded::no)
    return Succeeded::no;

  // build the psv-configured axes: Hessian-diagonal preconditioner + SVRG estimator
  {
    shared_ptr<HessianDiagonalPreconditioner<TargetT>> hd(new HessianDiagonalPreconditioner<TargetT>);
    hd->set_hessian_factor(this->precond_hessian_factor);
    hd->set_filter_fwhm_mm(this->precond_filter_fwhm_mm);
    hd->set_delta_rel(this->precond_delta_rel);
    hd->set_fov_mask_filename(this->fov_mask_filename);
    hd->set_fov_threshold(this->fov_threshold);
    hd->set_adjoint_ones_floor(this->adjoint_ones_floor);
    this->preconditioner_sptr = hd;

    shared_ptr<SVRGGradientEstimator<TargetT>> svrg(new SVRGGradientEstimator<TargetT>);
    svrg->set_complete_gradient_epoch_interval(this->complete_gradient_epoch_interval);
    this->gradient_estimator_sptr = svrg;
  }

  // hand the (pre-built) axes to the engine's shared component set-up
  return this->set_up_components(target_image_ptr);
}

END_NAMESPACE_STIR

///////// instantiations
#include "stir/DiscretisedDensity.h"
START_NAMESPACE_STIR
template class PreconditionedSVRGReconstruction<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
