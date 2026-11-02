//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_gradient_estimators_SVRGGradientEstimator_H__
#define __stir_recon_buildblock_gradient_estimators_SVRGGradientEstimator_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::SVRGGradientEstimator class

  \author Manil Sabeur

  Native port of the SVRG estimator of the PETRIC2 "psv" contribution (Hok Shing Wong,
  Margaret Duff, Matthias Ehrhardt, Georg Schramm).
*/

#include "stir/recon_buildblock/gradient_estimators/GeneralisedGradientEstimator.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
#include <vector>

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief Stochastic Variance Reduced Gradient estimator.

  \f[ \tilde g = M\left(\nabla f_i(x) - \nabla f_i(\tilde x)\right) + \sum_j \nabla f_j(\tilde x) \f]

  with \f$\tilde x\f$ a snapshot at which all subset gradients are recomputed at the start of every
  \c complete_gradient_epoch_interval epochs (the snapshot step itself uses the full gradient). This
  estimator owns that snapshot schedule.

  \par Parsing
  \verbatim
  SVRG Parameters :=
    complete gradient epoch interval := 2
  End SVRG Parameters :=
  \endverbatim
*/
template <typename TargetT>
class SVRGGradientEstimator
    : public RegisteredParsingObject<SVRGGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>>
{
  typedef RegisteredParsingObject<SVRGGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>> base_type;

public:
  static const char* const registered_name;

  SVRGGradientEstimator();

  Succeeded set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& gradient, const TargetT& current_estimate, int update_num, int subset_num, int num_subsets) override;

  //! setter used when the reconstruction constructs this estimator from its own keys
  void set_complete_gradient_epoch_interval(int v) { this->complete_gradient_epoch_interval = v; }

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  int complete_gradient_epoch_interval;

  /*! \brief the objective function, borrowed from the caller of set_up()
    \warning Non-owning: it points at the object passed to set_up(), whose lifetime must cover
    every call to compute(). Reset by each set_up().
  */
  GeneralisedObjectiveFunction<TargetT>* objective_ptr = nullptr;
  std::vector<shared_ptr<TargetT>> subset_gradients; // gradients at the snapshot
  shared_ptr<TargetT> summed_subset_gradients_sptr;  // their sum
};

END_NAMESPACE_STIR

#endif
