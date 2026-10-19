//
//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief implementation of the stir::IdentityPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/IdentityPreconditioner.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/info.h"

#include <algorithm>

START_NAMESPACE_STIR

template <typename TargetT>
const char* const IdentityPreconditioner<TargetT>::registered_name = "Identity";

template <typename TargetT>
IdentityPreconditioner<TargetT>::IdentityPreconditioner()
{
  set_defaults();
}

template <typename TargetT>
void
IdentityPreconditioner<TargetT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename TargetT>
void
IdentityPreconditioner<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Identity Preconditioner Parameters");
  this->parser.add_stop_key("End Identity Preconditioner Parameters");
}

template <typename TargetT>
Succeeded
IdentityPreconditioner<TargetT>::set_up(const GeneralisedObjectiveFunction<TargetT>&, const TargetT&)
{
  return Succeeded::yes;
}

template <typename TargetT>
void
IdentityPreconditioner<TargetT>::compute(TargetT& precond, const TargetT& /*current_estimate*/)
{
  info("recomputing preconditioner (Identity, P=1)");
  std::fill(precond.begin_all(), precond.end_all(), 1.F);
}

END_NAMESPACE_STIR

///////// instantiation
START_NAMESPACE_STIR
template class IdentityPreconditioner<DiscretisedDensity<3, float>>;
END_NAMESPACE_STIR
