//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_HessianDiagonalPreconditioner_H__
#define __stir_recon_buildblock_HessianDiagonalPreconditioner_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::HessianDiagonalPreconditioner class

  \author Manil Sabeur

  Native port of the preconditioner of the PETRIC2 "psv" contribution (Hok Shing Wong,
  Margaret Duff, Matthias Ehrhardt, Georg Schramm).
*/

#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief Preconditioner from the diagonal of the Hessian of the MAP objective.

  \f[ P = \mathrm{fov} \cdot \frac{x + \delta\, \max(x_\mathrm{sm})}{A^T 1 + c\, H_{jj}(x_\mathrm{sm})\, x} \f]

  where \f$A^T 1\f$ is the sensitivity (sum of the subset sensitivities), \f$H_{jj}\f$ the diagonal
  of the Hessian of the prior (requires <tt>prior.provides_Hessian_diagonal()</tt>), \f$x_\mathrm{sm}\f$
  a Gaussian-smoothed copy of the estimate, and \c fov a field-of-view mask. Without a prior this
  reduces to the EM preconditioner \f$x/A^T 1\f$.

  \par Parsing
  \verbatim
  Hessian Diagonal Preconditioner Parameters :=
    hessian factor := 2.0
    filter fwhm (mm) := 5.0
    delta rel := 0.0
    FOV mask filename :=
    FOV threshold := 1e-3
    adjoint ones floor := 1e-6
  End Hessian Diagonal Preconditioner Parameters :=
  \endverbatim
*/
template <typename TargetT>
class HessianDiagonalPreconditioner
    : public RegisteredParsingObject<HessianDiagonalPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>>
{
  typedef RegisteredParsingObject<HessianDiagonalPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>> base_type;

public:
  //! Name which will be used when parsing a preconditioner selection
  static const char* const registered_name;

  HessianDiagonalPreconditioner();

  Succeeded set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& precond, const TargetT& current_estimate) override;

  //! \name setters used when the reconstruction constructs this preconditioner from its own keys
  //@{
  void set_hessian_factor(double v) { this->hessian_factor = v; }
  void set_filter_fwhm_mm(double v) { this->filter_fwhm_mm = v; }
  void set_delta_rel(double v) { this->delta_rel = v; }
  void set_fov_threshold(double v) { this->fov_threshold = v; }
  void set_fov_mask_filename(const std::string& v) { this->fov_mask_filename = v; }
  void set_adjoint_ones_floor(double v) { this->adjoint_ones_floor = v; }
  //@}

protected:
  void set_defaults() override;
  void initialise_keymap() override;

private:
  double hessian_factor;
  double filter_fwhm_mm;
  double delta_rel;
  double fov_threshold;
  double adjoint_ones_floor;
  std::string fov_mask_filename;

  // set up in set_up()
  /*! \brief the objective function, borrowed from the caller of set_up()
    \warning Non-owning: it points at the object passed to set_up(), whose lifetime must cover
    every call to compute(). Reset by each set_up().
  */
  const GeneralisedObjectiveFunction<TargetT>* objective_ptr = nullptr;
  shared_ptr<TargetT> adjoint_ones_sptr;
  shared_ptr<TargetT> fov_mask_sptr;
  shared_ptr<DataProcessor<TargetT>> filter_sptr;
};

END_NAMESPACE_STIR

#endif
