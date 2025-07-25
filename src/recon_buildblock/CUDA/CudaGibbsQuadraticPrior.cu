/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup priors
  \ingroup CUDA
  \brief implementation of the stir::CudaGibbsQuadraticPrior class

  \author Matteo Colombo
  \author Kris Thielsmann
*/

#include "stir/recon_buildblock/GibbsQuadraticPrior.h"
#include "stir/BasicCoordinate.h"



START_NAMESPACE_STIR


// Implementation of constructors
template <typename elemT>
CudaGibbsQuadraticPrior<elemT>::CudaGibbsQuadraticPrior()
  : base_type() {}

template <typename elemT>
CudaGibbsQuadraticPrior<elemT>::CudaGibbsQuadraticPrior(const bool only_2D, float penalisation_factor)
  : base_type(only_2D, penalisation_factor) {}

// Explicit template instantiations
template class QuadraticPotential<float>;
template class CudaGibbsQuadraticPrior<float>;
// template class QuadraticPotential<double>;
// template class CudaGibbsQuadraticPrior<double>;

END_NAMESPACE_STIR
