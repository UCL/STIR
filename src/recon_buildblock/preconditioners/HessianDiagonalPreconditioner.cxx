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
  \brief implementation of the stir::HessianDiagonalPreconditioner class

  \author Manil Sabeur

  Native port of the preconditioner of the PETRIC2 "psv" contribution by Hok Shing Wong
  (Univ. of Bath), Margaret Duff (STFC), Matthias Ehrhardt (Univ. of Bath) and Georg Schramm
  (KU Leuven).
*/

#include "stir/recon_buildblock/preconditioners/HessianDiagonalPreconditioner.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/format.h"
#include "stir/error.h"
#include "stir/unique_ptr.h"

#include <algorithm>
#include "boost/lambda/lambda.hpp"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const HessianDiagonalPreconditioner<TargetT>::registered_name = "Hessian Diagonal";

template <typename TargetT>
HessianDiagonalPreconditioner<TargetT>::HessianDiagonalPreconditioner()
{
  set_defaults();
}

template <typename TargetT>
void
HessianDiagonalPreconditioner<TargetT>::set_defaults()
{
  base_type::set_defaults();
  // defaults of the original PETRIC2 "psv" submission
  this->hessian_factor = 2.0;
  this->filter_fwhm_mm = 5.0;
  this->delta_rel = 0.0;
  this->fov_threshold = 1e-3;
  this->adjoint_ones_floor = 1e-6;
  this->fov_mask_filename = "";
}

template <typename TargetT>
void
HessianDiagonalPreconditioner<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Hessian Diagonal Preconditioner Parameters");
  this->parser.add_key("hessian factor", &this->hessian_factor);
  this->parser.add_key("filter fwhm (mm)", &this->filter_fwhm_mm);
  this->parser.add_key("delta rel", &this->delta_rel);
  this->parser.add_key("FOV mask filename", &this->fov_mask_filename);
  this->parser.add_key("FOV threshold", &this->fov_threshold);
  this->parser.add_key("adjoint ones floor", &this->adjoint_ones_floor);
  this->parser.add_stop_key("End Hessian Diagonal Preconditioner Parameters");
}

template <typename TargetT>
Succeeded
HessianDiagonalPreconditioner<TargetT>::set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function,
                                               const TargetT& target)
{
  // A^T 1 (the sensitivity) is Poisson-specific
  const PoissonLogLikelihoodWithLinearModelForMean<TargetT>* const poisson_obj_ptr
      = dynamic_cast<const PoissonLogLikelihoodWithLinearModelForMean<TargetT>*>(&objective_function);
  if (is_null_ptr(poisson_obj_ptr))
    {
      warning("Hessian Diagonal preconditioner: objective function must be of a type derived from "
              "PoissonLogLikelihoodWithLinearModelForMean (needs the subset sensitivities)\n");
      return Succeeded::no;
    }

  // the prior must provide its Hessian diagonal. Refuse here, at set-up, rather than crashing
  // mid-reconstruction on the default compute_Hessian_diagonal(), which only calls error().
  if (!objective_function.prior_is_zero())
    {
      const GeneralisedPrior<TargetT>& prior = *objective_function.get_prior_ptr();
      if (!prior.provides_Hessian_diagonal())
        {
          warning(format("Hessian Diagonal preconditioner: prior '{}' does not implement "
                         "compute_Hessian_diagonal(). Compatible priors: Gibbs Relative Difference, "
                         "Gibbs Quadratic.",
                         objective_function.get_prior_ptr()->get_registered_name()));
          return Succeeded::no;
        }
    }
  this->objective_ptr = &objective_function;

  const int num_subsets = objective_function.get_num_subsets();

  // A^T 1 = sum of the subset sensitivities
  this->adjoint_ones_sptr.reset(target.get_empty_copy());
  std::fill(this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), 0.F);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    std::transform(this->adjoint_ones_sptr->begin_all(),
                   this->adjoint_ones_sptr->end_all(),
                   poisson_obj_ptr->get_subset_sensitivity(subset_num).begin_all_const(),
                   this->adjoint_ones_sptr->begin_all(),
                   _1 + _2);

  // FOV mask, from file or by thresholding A^T 1: outside the FOV A^T 1 is close to 0 and P would blow up
  this->fov_mask_sptr.reset(target.get_empty_copy());
  if (this->fov_mask_filename != "")
    {
      shared_ptr<TargetT> mask_sptr = read_from_file<TargetT>(this->fov_mask_filename);
      std::string explanation;
      if (!mask_sptr->has_same_characteristics(target, explanation))
        {
          warning(format("Hessian Diagonal preconditioner: FOV mask should have the same characteristics as "
                         "the target image: {}",
                         explanation));
          return Succeeded::no;
        }
      *this->fov_mask_sptr = *mask_sptr;
    }
  else
    {
      const float threshold = static_cast<float>(this->fov_threshold)
                              * (*std::max_element(this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all()));
      std::transform(this->adjoint_ones_sptr->begin_all_const(),
                     this->adjoint_ones_sptr->end_all_const(),
                     this->fov_mask_sptr->begin_all(),
                     [threshold](float a) { return a > threshold ? 1.F : 0.F; });
    }

  // floor A^T 1 to avoid division by zero
  threshold_min_to_small_positive_value(
      this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), static_cast<float>(this->adjoint_ones_floor));

  // Gaussian filter used before evaluating the Hessian diagonal
  {
    shared_ptr<SeparableGaussianImageFilter<float>> filter_sptr(new SeparableGaussianImageFilter<float>);
    BasicCoordinate<3, float> fwhms;
    fwhms[1] = fwhms[2] = fwhms[3] = static_cast<float>(this->filter_fwhm_mm);
    filter_sptr->set_fwhms(fwhms);
    if (filter_sptr->set_up(target) == Succeeded::no)
      {
        warning("Hessian Diagonal preconditioner: could not set up the Gaussian filter\n");
        return Succeeded::no;
      }
    this->filter_sptr = filter_sptr;
  }

  return Succeeded::yes;
}

template <typename TargetT>
void
HessianDiagonalPreconditioner<TargetT>::compute(TargetT& precond, const TargetT& current_image_estimate)
{
  info("recomputing preconditioner (Hessian diagonal)");

  // numerator = mask * (x + delta_rel * max(x_smoothed))
  unique_ptr<TargetT> numerator_ptr(current_image_estimate.clone());
  if (this->delta_rel != 0)
    {
      unique_ptr<TargetT> smoothed_ptr(current_image_estimate.clone());
      this->filter_sptr->apply(*smoothed_ptr);
      const float bump
          = static_cast<float>(this->delta_rel) * (*std::max_element(smoothed_ptr->begin_all(), smoothed_ptr->end_all()));
      std::transform(numerator_ptr->begin_all(), numerator_ptr->end_all(), numerator_ptr->begin_all(), _1 + bump);
    }
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 this->fov_mask_sptr->begin_all(),
                 numerator_ptr->begin_all(),
                 _1 * _2);

  // denominator = A^T 1 + c * H_jj(x_smoothed) * x
  unique_ptr<TargetT> denominator_ptr(current_image_estimate.get_empty_copy());
  if (is_null_ptr(this->objective_ptr))
    error("Hessian Diagonal preconditioner: set_up() has to be called before compute()");
  if (this->objective_ptr->prior_is_zero())
    {
      // H_jj = 0 without a prior: reduces to the MLEM preconditioner
      *denominator_ptr = *this->adjoint_ones_sptr;
    }
  else
    {
      unique_ptr<TargetT> smoothed_ptr(current_image_estimate.clone());
      this->filter_sptr->apply(*smoothed_ptr); // H_jj is evaluated on the smoothed image
      this->objective_ptr->get_prior_ptr()->compute_Hessian_diagonal(*denominator_ptr, *smoothed_ptr);
      const float factor = static_cast<float>(this->hessian_factor);
      // c * H_jj * x, then + A^T 1, with x the unsmoothed image
      std::transform(denominator_ptr->begin_all(),
                     denominator_ptr->end_all(),
                     current_image_estimate.begin_all_const(),
                     denominator_ptr->begin_all(),
                     _1 * _2);
      std::transform(denominator_ptr->begin_all(),
                     denominator_ptr->end_all(),
                     this->adjoint_ones_sptr->begin_all(),
                     denominator_ptr->begin_all(),
                     _1 * factor + _2);
    }
  threshold_min_to_small_positive_value(
      denominator_ptr->begin_all(), denominator_ptr->end_all(), static_cast<float>(this->adjoint_ones_floor));

  std::transform(
      numerator_ptr->begin_all(), numerator_ptr->end_all(), denominator_ptr->begin_all(), precond.begin_all(), _1 / _2);
}

END_NAMESPACE_STIR

///////// instantiation
#include "stir/DiscretisedDensity.h"
START_NAMESPACE_STIR
template class HessianDiagonalPreconditioner<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
