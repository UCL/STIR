//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_gradient_estimators_FullGradientEstimator_H__
#define __stir_recon_buildblock_gradient_estimators_FullGradientEstimator_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::FullGradientEstimator class
  \author Manil Sabeur
*/

#include "stir/recon_buildblock/gradient_estimators/GeneralisedGradientEstimator.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief The exact (full) gradient \f$ \tilde g = \sum_i \nabla f_i(x) = \nabla\Phi(x) \f$.

  Deterministic, zero variance, but one full data pass per update. Combined with an EM
  preconditioner and unit step this reproduces MLEM. Mostly a reference / building block.

  \par Parsing
  \verbatim
  Full Gradient Parameters :=
  End Full Gradient Parameters :=
  \endverbatim
*/
template <typename TargetT>
class FullGradientEstimator
    : public RegisteredParsingObject<FullGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>>
{
  typedef RegisteredParsingObject<FullGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>> base_type;

public:
  static const char* const registered_name;

  FullGradientEstimator();

  Succeeded set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& gradient, const TargetT& current_estimate, int update_num, int subset_num, int num_subsets) override;

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  /*! \brief the objective function, borrowed from the caller of set_up()
    \warning Non-owning: it points at the object passed to set_up(), whose lifetime must cover
    every call to compute(). Reset by each set_up().
  */
  GeneralisedObjectiveFunction<TargetT>* objective_ptr = nullptr;
  shared_ptr<TargetT> scratch_sptr;
};

END_NAMESPACE_STIR

#endif
