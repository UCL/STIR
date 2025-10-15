//
//
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief  implementation of the stir::GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/error.h"

START_NAMESPACE_STIR

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif


template class GeneralisedPrior<DiscretisedDensity<3, float>>;
template class GeneralisedPrior<ParametricVoxelsOnCartesianGrid>;


END_NAMESPACE_STIR
