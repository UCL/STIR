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
  \brief Implementation of class stir::CudaGibbsRelativeDifferencePrior

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsRelativeDifferencePrior.h"
#include "stir/BasicCoordinate.h" 

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
CudaGibbsRelativeDifferencePrior<elemT>::CudaGibbsRelativeDifferencePrior()
{
  set_defaults();
}

template <typename elemT>
CudaGibbsRelativeDifferencePrior<elemT>::CudaGibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor,float gamma_v, float epsilon_v)
  : base_type(only_2D, penalisation_factor) {this->potential.gamma = gamma_v; this->potential.epsilon = epsilon_v;}

template <typename elemT>
void
CudaGibbsRelativeDifferencePrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->potential.gamma = 2;
  this->potential.epsilon = 1e-7F;
}

template <typename elemT>
bool
CudaGibbsRelativeDifferencePrior<elemT>::is_convex() const
{
  return true;
}

template <typename elemT>
std::string 
CudaGibbsRelativeDifferencePrior<elemT>::get_parsing_name() const
{
  return registered_name;
}

// 
template class CudaGibbsRelativeDifferencePrior<float>;

END_NAMESPACE_STIR
