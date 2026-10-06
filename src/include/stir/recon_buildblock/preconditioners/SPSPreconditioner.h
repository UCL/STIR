//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_SPSPreconditioner_H__
#define __stir_recon_buildblock_SPSPreconditioner_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::SPSPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;
template <typename TargetT>
class PriorWithParabolicSurrogate;

/*!
  \ingroup recon_buildblock
  \brief The separable-paraboloidal-surrogate (SPS) preconditioner
  \f$ P = 1 / (2\,S_\mathrm{prior}(x) + d_\mathrm{lik}) \f$.

  \f$d_\mathrm{lik}\f$ is the (sign-flipped) row-sum of the approximate Hessian of the log-likelihood,
  precomputed once at set_up via
  GeneralisedObjectiveFunction::add_multiplication_with_approximate_Hessian_without_penalty. When a
  prior is present, \f$S_\mathrm{prior}(x)\f$ is its parabolic-surrogate curvature, recomputed at each
  \c compute() call. Composed with the \c "Ordered Subsets" estimator (numerator \f$M\nabla f_s\f$) and
  the per-epoch relaxation step, this reproduces STIR's native OSSPS.

  \par Capability contract
  Requires the prior (if any) to be a stir::PriorWithParabolicSurrogate; otherwise \c set_up refuses
  (\c Succeeded::no). Currently this means the \c "Quadratic" or \c "Logcosh" prior, or no prior.
  No implementation of the Relative Difference Prior (neither the Gibbs one nor the older
  \c "Relative Difference Prior") provides a parabolic surrogate, so SPS cannot be used with an RDP.

  Subset-independent (full-scale curvature); pair with a full-gradient-scale gradient. \c P depends on
  \a x, so the reconstruction schedule should recompute it every subiteration.

  \par Parsing
  \verbatim
  SPS Preconditioner Parameters :=
    denominator floor := 1e-6
  End SPS Preconditioner Parameters :=
  \endverbatim
*/
template <typename TargetT>
class SPSPreconditioner : public RegisteredParsingObject<SPSPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>>
{
  typedef RegisteredParsingObject<SPSPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>> base_type;

public:
  static const char* const registered_name;

  SPSPreconditioner();

  Succeeded set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& precond, const TargetT& current_estimate) override;

  void set_denominator_floor(double v) { this->denominator_floor = v; }

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  double denominator_floor;

  //! (sign-flipped) row-sum of the approximate Hessian of the likelihood, computed once at set_up
  shared_ptr<TargetT> precomputed_denominator_sptr;
  /*! \brief parabolic-surrogate prior, borrowed from the objective (null if there is no prior)
    \warning Non-owning: it points into the objective function passed to set_up(), whose lifetime
    must cover every call to compute(). Reset by each set_up().
  */
  PriorWithParabolicSurrogate<TargetT>* surrogate_prior_ptr = nullptr;
};

END_NAMESPACE_STIR

#endif
