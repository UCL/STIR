//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_gradient_estimators_GeneralisedGradientEstimator_H__
#define __stir_recon_buildblock_gradient_estimators_GeneralisedGradientEstimator_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::GeneralisedGradientEstimator class

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
  \brief Base class for gradient estimators of a preconditioned gradient reconstruction.

  Produces \f$\tilde g\f$, the estimate of \f$\nabla\Phi\f$ used at each update. It is a registered,
  \c .par-driven family: the reconstruction selects a concrete estimator by a
  <tt>gradient estimator type :=</tt> key. Members trade cost against variance (Full / Ordered
  Subsets / SVRG / SAGA).

  Unlike a preconditioner, an estimator may carry \em state (stored subset gradients) and its own
  \em schedule (e.g. SVRG's snapshot every few epochs): \c compute() therefore receives the
  iteration context, and each estimator owns whatever schedule it needs.
*/
template <typename TargetT>
class GeneralisedGradientEstimator : public RegisteredObject<GeneralisedGradientEstimator<TargetT>>
{
public:
  ~GeneralisedGradientEstimator() override {}

  //! set up internal state from the objective function and a model image
  /*! Non-const objective: compute_sub_gradient() is not const. */
  virtual Succeeded set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) = 0;

  //! write the gradient estimate at \a current_estimate into \a gradient (pre-allocated)
  /*! \param update_num  global update counter (0-based)
      \param subset_num  the subset to process for this update
      \param num_subsets total number of subsets */
  virtual void compute(TargetT& gradient, const TargetT& current_estimate, int update_num, int subset_num, int num_subsets) = 0;

  //! whether the output is the M-scaled gradient of the current subset (Ordered Subsets)
  /*! Scale-compatibility contract: a subset-scaled preconditioner (e.g. EM with
      'use subset sensitivity', whose \f$P_s\f$ uses the current subset's sensitivity) is only exact
      against this per-subset output. Full-gradient estimators (Full, SVRG, SAGA: their output
      estimates \f$\nabla\Phi\f$ as a whole) return false and need a full-scale preconditioner. */
  virtual bool is_subset_gradient() const { return false; }
};

END_NAMESPACE_STIR

#endif
