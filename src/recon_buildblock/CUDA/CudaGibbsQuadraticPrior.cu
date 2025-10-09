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
void
CudaGibbsQuadraticPrior<elemT>::initialise_keymap()
{
  this->parser.add_start_key("Cuda Gibbs Quadratic Prior Parameters");
  base_type::initialise_keymap();
  this->parser.add_stop_key("END Cuda Gibbs Quadratic Prior Parameters");
}

template class CudaGibbsQuadraticPrior<float>;

END_NAMESPACE_STIR
