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
  \brief Implementation of class stir::CudaGibbsQuadraticPrior

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/


#include "stir/recon_buildblock/GibbsQuadraticPrior.h"


START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
CudaGibbsQuadraticPrior<elemT>::CudaGibbsQuadraticPrior()
{
  set_defaults();
}

template <typename elemT>
CudaGibbsQuadraticPrior<elemT>::CudaGibbsQuadraticPrior(const bool only_2D, float penalisation_factor)
  : base_type(only_2D, penalisation_factor) {}

template <typename elemT>
void
CudaGibbsQuadraticPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
bool
CudaGibbsQuadraticPrior<elemT>::is_convex() const
{
  return true;
}

template <typename elemT>
std::string 
CudaGibbsQuadraticPrior<elemT>::get_parsing_name() const
{
  return registered_name;
}

template class CudaGibbsQuadraticPrior<float>;

END_NAMESPACE_STIR
