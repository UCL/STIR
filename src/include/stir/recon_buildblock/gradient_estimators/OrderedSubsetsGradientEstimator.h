//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_gradient_estimators_OrderedSubsetsGradientEstimator_H__
#define __stir_recon_buildblock_gradient_estimators_OrderedSubsetsGradientEstimator_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::OrderedSubsetsGradientEstimator class
  \author Manil Sabeur
*/

#include "stir/recon_buildblock/gradient_estimators/GeneralisedGradientEstimator.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief Plain ordered-subsets gradient estimate \f$ \tilde g = M\,\nabla f_i(x) \f$.

  One subset per update, scaled by the number of subsets \f$M\f$. Unbiased, but without variance
  reduction (compare SVRGGradientEstimator and SAGAGradientEstimator).

  \par Parsing
  \verbatim
  Ordered Subsets Parameters :=
  End Ordered Subsets Parameters :=
  \endverbatim
*/
template <typename TargetT>
class OrderedSubsetsGradientEstimator
    : public RegisteredParsingObject<OrderedSubsetsGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>>
{
  typedef RegisteredParsingObject<OrderedSubsetsGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>> base_type;

public:
  static const char* const registered_name;

  OrderedSubsetsGradientEstimator();

  Succeeded set_up(GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& gradient, const TargetT& current_estimate, int update_num, int subset_num, int num_subsets) override;

  //! the output is the M-scaled current-subset gradient, which pairs exactly with subset-scaled preconditioners
  bool is_subset_gradient() const override { return true; }

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  /*! \brief the objective function, borrowed from the caller of set_up()
    \warning Non-owning: it points at the object passed to set_up(), whose lifetime must cover
    every call to compute(). Reset by each set_up().
  */
  GeneralisedObjectiveFunction<TargetT>* objective_ptr = nullptr;
};

END_NAMESPACE_STIR

#endif
