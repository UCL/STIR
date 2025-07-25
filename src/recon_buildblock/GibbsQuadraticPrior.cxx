//
//
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/

/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief Instantiations of class stir::CudaGibbsRelativeDifferencePrior

  \author Matteo Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsQuadraticPrior.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
GibbsQuadraticPrior<elemT>::GibbsQuadraticPrior()
  : base_type() {}

template <typename elemT>
GibbsQuadraticPrior<elemT>::GibbsQuadraticPrior(const bool only_2D, float penalisation_factor)
  : base_type(only_2D, penalisation_factor) {}

// Explicit template instantiations
template class QuadraticPotential<float>;
template class GibbsQuadraticPrior<float>;

// template class QuadraticPotential<double>;
// template class GibbsQuadraticPrior<double>;

END_NAMESPACE_STIR