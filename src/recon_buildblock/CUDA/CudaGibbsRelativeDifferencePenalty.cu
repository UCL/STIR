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
  \ingroup CUDA
  \brief Implementation of class stir::CudaGibbsRelativeDifferencePenalty

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsRelativeDifferencePenalty.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
CudaGibbsRelativeDifferencePenalty<elemT>::CudaGibbsRelativeDifferencePenalty()
{
  this->set_defaults();
}

template <typename elemT>
CudaGibbsRelativeDifferencePenalty<elemT>::CudaGibbsRelativeDifferencePenalty(const bool only_2D,
                                                                              float penalisation_factor,
                                                                              float gamma_v,
                                                                              float epsilon_v)
    : base_type(only_2D, penalisation_factor)
{
  this->potential.gamma = gamma_v;
  this->potential.epsilon = epsilon_v;
}

template class CudaGibbsRelativeDifferencePenalty<float>;

END_NAMESPACE_STIR
