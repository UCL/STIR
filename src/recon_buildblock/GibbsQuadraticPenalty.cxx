//
//
/*
    Copyright (C) 2025, University College London
    Copyright (C) 2025, University of Milano-Bicocca
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup priors
  \brief Implementation of class stir::GibbsQuadraticPenalty

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsQuadraticPenalty.h"

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
GibbsQuadraticPenalty<elemT>::GibbsQuadraticPenalty()
{
  this->set_defaults();
}

template <typename elemT>
GibbsQuadraticPenalty<elemT>::GibbsQuadraticPenalty(const bool only_2D, float penalisation_factor)
    : base_type(only_2D, penalisation_factor)
{}

// Explicit template instantiations
template class GibbsQuadraticPenalty<float>;

END_NAMESPACE_STIR
