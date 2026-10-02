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
  \brief Instantiations of the stir::GeneralisedPreconditioner class

  \author Manil Sabeur
*/

#include "stir/recon_buildblock/preconditioners/GeneralisedPreconditioner.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template class GeneralisedPreconditioner<DiscretisedDensity<3, float>>;

END_NAMESPACE_STIR
