//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_GeneralisedPreconditioner_H__
#define __stir_recon_buildblock_GeneralisedPreconditioner_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::GeneralisedPreconditioner class

  \author Manil Sabeur
*/

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief Base class for image-space preconditioners of a preconditioned gradient reconstruction.

  A preconditioner produces, from the current image estimate, an image \c P that is multiplied
  (element-wise) with the gradient before the step. It is a registered, \c .par-driven family: the
  reconstruction selects a concrete preconditioner by a <tt>preconditioner type :=</tt> key.

  The reconstruction owns the \em schedule (when to recompute \c P); a preconditioner only answers
  "given the current estimate, produce \c P".
*/
template <typename TargetT>
class GeneralisedPreconditioner : public RegisteredObject<GeneralisedPreconditioner<TargetT>>
{
public:
  ~GeneralisedPreconditioner() override {}

  //! set up internal quantities from the objective function and a model image
  /*! Called once, after the objective function has been set up. Returns Succeeded::no (with a
      warning) if the preconditioner cannot be used with this objective function or prior. */
  virtual Succeeded set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) = 0;

  //! compute the preconditioner image at \a current_estimate, writing it into \a precond
  /*! \a precond must already be allocated with the correct characteristics. */
  virtual void compute(TargetT& precond, const TargetT& current_estimate) = 0;

  //! compute the preconditioner image at \a current_estimate for the current subset
  /*! The default implementation ignores the subset and forwards to the subset-independent
      compute(). Only preconditioners that depend on the subset override it, e.g. \c EM with
      <tt>use subset sensitivity</tt>, which uses the subset sensitivity \f$A_s^T 1\f$ (OSEM). */
  virtual void compute(TargetT& precond, const TargetT& current_estimate, int subset_num, int num_subsets)
  {
    compute(precond, current_estimate);
  }

  //! whether \f$P\f$ is scaled to the current subset (e.g. EM with 'use subset sensitivity')
  /*! Scale-compatibility contract (checked by the reconstruction at set-up): a subset-scaled
      preconditioner is only exact against the per-subset output of the 'Ordered Subsets'
      estimator; composing it with a full-gradient estimator is refused. */
  virtual bool is_subset_scaled() const { return false; }
};

END_NAMESPACE_STIR

#endif
