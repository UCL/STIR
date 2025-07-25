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

#include "stir/recon_buildblock/GibbsRelativeDifferencePrior.h"
#include "stir/BasicCoordinate.h" 

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
CudaGibbsRelativeDifferencePrior<elemT>::CudaGibbsRelativeDifferencePrior()
  : base_type() {this->potential.gamma = 2.0f; this->potential.epsilon = 0.0f;}

template <typename elemT>
CudaGibbsRelativeDifferencePrior<elemT>::CudaGibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v)
  : base_type(only_2D, penalisation_factor) {this->potential.gamma = gamma_v; this->potential.epsilon = epsilon_v;}

// Explicit template instantiations
//template class RelativeDifferencePotential<float>;
template class CudaGibbsRelativeDifferencePrior<float>;

// template class RelativeDifferencePotential<double>;
// template class CudaGibbsRelativeDifferencePrior<double>;

END_NAMESPACE_STIR
