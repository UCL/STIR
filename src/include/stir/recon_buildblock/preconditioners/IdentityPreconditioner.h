//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_IdentityPreconditioner_H__
#define __stir_recon_buildblock_IdentityPreconditioner_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of the stir::IdentityPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <typename TargetT>
class GeneralisedObjectiveFunction;

/*!
  \ingroup recon_buildblock
  \brief The trivial preconditioner \f$ P = 1 \f$ (no preconditioning).

  The update becomes plain projected gradient ascent. It usually needs a much smaller step size
  than the other preconditioners, and is mainly useful for comparison.

  \par Parsing
  \verbatim
  Identity Preconditioner Parameters :=
  End Identity Preconditioner Parameters :=
  \endverbatim
*/
template <typename TargetT>
class IdentityPreconditioner : public RegisteredParsingObject<IdentityPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>>
{
  typedef RegisteredParsingObject<IdentityPreconditioner<TargetT>, GeneralisedPreconditioner<TargetT>> base_type;

public:
  static const char* const registered_name;

  IdentityPreconditioner();

  Succeeded set_up(const GeneralisedObjectiveFunction<TargetT>& objective_function, const TargetT& target) override;
  void compute(TargetT& precond, const TargetT& current_estimate) override;

protected:
  void set_defaults() override;
  void initialise_keymap() override;
};

END_NAMESPACE_STIR

#endif
