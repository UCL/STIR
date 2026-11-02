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
  \brief implementation of the stir::EMPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/EMPreconditioner.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/thresholding.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/format.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/unique_ptr.h"

#include <algorithm>
#include "boost/lambda/lambda.hpp"

using boost::lambda::_1;
using boost::lambda::_2;

START_NAMESPACE_STIR

template <typename TargetT>
const char* const EMPreconditioner<TargetT>::registered_name = "EM";

template <typename TargetT>
EMPreconditioner<TargetT>::EMPreconditioner()
{
  set_defaults();
}

template <typename TargetT>
void
EMPreconditioner<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->fov_threshold = 1e-3;
  this->adjoint_ones_floor = 1e-6;
  this->fov_mask_filename = "";
  this->use_subset_sensitivity = false;
}

template <typename TargetT>
void
EMPreconditioner<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("EM Preconditioner Parameters");
  this->parser.add_key("FOV mask filename", &this->fov_mask_filename);
  this->parser.add_key("FOV threshold", &this->fov_threshold);
  this->parser.add_key("adjoint ones floor", &this->adjoint_ones_floor);
  this->parser.add_key("use subset sensitivity", &this->use_subset_sensitivity);
  this->parser.add_stop_key("End EM Preconditioner Parameters");
}

template <typename TargetT>
Succeeded
EMPreconditioner<TargetT>::set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target)
{
  const PoissonLogLikelihoodWithLinearModelForMean<TargetT>* const poisson_obj_ptr
      = dynamic_cast<const PoissonLogLikelihoodWithLinearModelForMean<TargetT>*>(&objective_function);
  if (is_null_ptr(poisson_obj_ptr))
    {
      warning("EM preconditioner: objective function must be of a type derived from "
              "PoissonLogLikelihoodWithLinearModelForMean (needs the subset sensitivities)\n");
      return Succeeded::no;
    }

  const int num_subsets = objective_function.get_num_subsets();

  this->adjoint_ones_sptr.reset(target.get_empty_copy());
  std::fill(this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), 0.F);
  for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
    std::transform(this->adjoint_ones_sptr->begin_all(),
                   this->adjoint_ones_sptr->end_all(),
                   poisson_obj_ptr->get_subset_sensitivity(subset_num).begin_all_const(),
                   this->adjoint_ones_sptr->begin_all(),
                   _1 + _2);

  this->fov_mask_sptr.reset(target.get_empty_copy());
  if (this->fov_mask_filename != "")
    {
      shared_ptr<TargetT> mask_sptr = read_from_file<TargetT>(this->fov_mask_filename);
      std::string explanation;
      if (!mask_sptr->has_same_characteristics(target, explanation))
        {
          warning(format("EM preconditioner: FOV mask should have the same characteristics as the target "
                         "image: {}",
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

  threshold_min_to_small_positive_value(
      this->adjoint_ones_sptr->begin_all(), this->adjoint_ones_sptr->end_all(), static_cast<float>(this->adjoint_ones_floor));

  // OSEM mode: store the per-subset sensitivities A_s^T 1, floored and pre-multiplied by M, so that
  // compute() can use a plain x/denom. The M cancels the M that OrderedSubsetsGradientEstimator
  // applies to the subset gradient (M*grad_s), giving x*grad_s/(A_s^T1) = the OSEM step.
  if (this->use_subset_sensitivity)
    {
      const float M = static_cast<float>(num_subsets);
      this->subset_adjoint_ones_sptr.resize(num_subsets);
      for (int subset_num = 0; subset_num < num_subsets; ++subset_num)
        {
          this->subset_adjoint_ones_sptr[subset_num].reset(poisson_obj_ptr->get_subset_sensitivity(subset_num).clone());
          threshold_min_to_small_positive_value(this->subset_adjoint_ones_sptr[subset_num]->begin_all(),
                                                this->subset_adjoint_ones_sptr[subset_num]->end_all(),
                                                static_cast<float>(this->adjoint_ones_floor));
          std::transform(this->subset_adjoint_ones_sptr[subset_num]->begin_all(),
                         this->subset_adjoint_ones_sptr[subset_num]->end_all(),
                         this->subset_adjoint_ones_sptr[subset_num]->begin_all(),
                         _1 * M);
        }
    }

  return Succeeded::yes;
}

template <typename TargetT>
void
EMPreconditioner<TargetT>::compute(TargetT& precond, const TargetT& current_estimate)
{
  info("recomputing preconditioner (EM)");
  // P = mask * x / A^T 1
  unique_ptr<TargetT> numerator_ptr(current_estimate.clone());
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 this->fov_mask_sptr->begin_all(),
                 numerator_ptr->begin_all(),
                 _1 * _2);
  std::transform(
      numerator_ptr->begin_all(), numerator_ptr->end_all(), this->adjoint_ones_sptr->begin_all(), precond.begin_all(), _1 / _2);
}

template <typename TargetT>
void
EMPreconditioner<TargetT>::compute(TargetT& precond, const TargetT& current_estimate, int subset_num, int num_subsets)
{
  if (!this->use_subset_sensitivity)
    {
      // subset-independent behaviour (MLEM / SVRG): full A^T 1
      this->compute(precond, current_estimate);
      return;
    }
  // invariant: set_up() sized the table from the objective function's number of subsets, which
  // IterativeReconstruction::set_up() has already reconciled with the reconstruction's own. Check
  // it once here rather than indexing blindly below.
  if (subset_num < 0 || subset_num >= static_cast<int>(this->subset_adjoint_ones_sptr.size())
      || num_subsets != static_cast<int>(this->subset_adjoint_ones_sptr.size()))
    error(format("EM preconditioner: compute() called for subset {} of {}, but set_up() prepared {} subset "
                 "sensitivities",
                 subset_num,
                 num_subsets,
                 this->subset_adjoint_ones_sptr.size()));
  info(format("recomputing preconditioner (EM, subset {})", subset_num));
  // P_s = mask * x / (M * A_s^T 1)   (denominator already scaled by M at set_up)
  unique_ptr<TargetT> numerator_ptr(current_estimate.clone());
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 this->fov_mask_sptr->begin_all(),
                 numerator_ptr->begin_all(),
                 _1 * _2);
  std::transform(numerator_ptr->begin_all(),
                 numerator_ptr->end_all(),
                 this->subset_adjoint_ones_sptr[subset_num]->begin_all(),
                 precond.begin_all(),
                 _1 / _2);
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class EMPreconditioner<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
