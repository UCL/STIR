//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_EMPreconditioner_H__
#define __stir_recon_buildblock_EMPreconditioner_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::EMPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
#include <string>
#include <vector>

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief The classic EM preconditioner \f$ P = \mathrm{fov}\cdot x / A^T 1 \f$.

  Ignores the prior entirely. This is the limit of the Hessian-diagonal preconditioner without a
  prior. With a unit step it turns the Full Gradient and Ordered Subsets estimators into MLEM and
  OSEM respectively (the latter with <tt>use subset sensitivity</tt>).

  \par Parsing
  \verbatim
  EM Preconditioner Parameters :=
    FOV mask filename :=
    FOV threshold := 1e-3
    adjoint ones floor := 1e-6
    use subset sensitivity := 0
  End EM Preconditioner Parameters :=
  \endverbatim
*/
template <typename TargetT>
class EMPreconditioner : public RegisteredParsingObject<EMPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>>
{
  typedef RegisteredParsingObject<EMPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>> base_type;

public:
  static const char* const registered_name;

  EMPreconditioner();

  Succeeded set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& precond, const TargetT& current_estimate) override;
  //! subset-aware variant: \f$ P_s = \mathrm{fov}\cdot x / (M \cdot A_s^T 1) \f$ (reproduces OSEM)
  /*! The factor \f$M = \f$ \a num_subsets cancels the \f$M\f$ that OrderedSubsetsGradientEstimator
      applies to the subset gradient, so \f$P_s\cdot(M\nabla f_s) = x\cdot\nabla f_s/A_s^T1\f$. When
      \c use_subset_sensitivity is false this forwards to the subset-independent compute(). */
  void compute(TargetT& precond, const TargetT& current_estimate, int subset_num, int num_subsets) override;

  //! subset-scaled exactly when 'use subset sensitivity' is on (the OSEM mode)
  bool is_subset_scaled() const override { return this->use_subset_sensitivity; }

  void set_fov_threshold(double v) { this->fov_threshold = v; }
  void set_fov_mask_filename(const std::string& v) { this->fov_mask_filename = v; }
  void set_adjoint_ones_floor(double v) { this->adjoint_ones_floor = v; }
  void set_use_subset_sensitivity(bool v) { this->use_subset_sensitivity = v; }

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  double fov_threshold;
  double adjoint_ones_floor;
  std::string fov_mask_filename;
  //! use per-subset sensitivity \f$A_s^T1\f$ (OSEM) instead of the full \f$A^T1\f$ (MLEM/SVRG)
  bool use_subset_sensitivity;

  shared_ptr<TargetT> adjoint_ones_sptr;
  shared_ptr<TargetT> fov_mask_sptr;
  //! per-subset floored sensitivities \f$A_s^T1\f$, filled at set_up when \c use_subset_sensitivity
  std::vector<shared_ptr<TargetT>> subset_adjoint_ones_sptr;
};

END_NAMESPACE_STIR

#endif
