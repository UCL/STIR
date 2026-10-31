//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_gradient_estimators_SAGAGradientEstimator_H__
#define __stir_recon_buildblock_gradient_estimators_SAGAGradientEstimator_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::SAGAGradientEstimator class
  \author Manil Sabeur
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
  \brief SAGA variance-reduced gradient estimate.

  \f[ \tilde g = M\left(\nabla f_i(x) - g_i\right) + \sum_j g_j \f]

  where \f$g_i\f$ is a per-subset gradient table, refreshed \em incrementally (one entry per update),
  unlike SVRG which refreshes all entries at a periodic snapshot. The table (\f$M\f$ images) is
  initialised by a single full pass on the first update. Same memory footprint as SVRG's snapshot
  state, but no periodic full pass.

  \par Parsing
  \verbatim
  SAGA Parameters :=
  End SAGA Parameters :=
  \endverbatim
*/
template <typename TargetT>
class SAGAGradientEstimator
    : public RegisteredParsingObject<SAGAGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>>
{
  typedef RegisteredParsingObject<SAGAGradientEstimator<TargetT>, GeneralisedGradientEstimator<TargetT>> base_type;

public:
  static const char* const registered_name;

  SAGAGradientEstimator();

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
  std::vector<shared_ptr<TargetT>> table; // g_i, one per subset
  shared_ptr<TargetT> summed_table_sptr;  // running sum of the table
  shared_ptr<TargetT> scratch_sptr;       // new subset gradient
  shared_ptr<TargetT> delta_sptr;         // new subset gradient - old table entry
  bool initialised = false;
};

END_NAMESPACE_STIR

#endif
