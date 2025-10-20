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
  \brief Implementation of class stir::CudaGibbsQuadraticPenalty

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/


#include "stir/recon_buildblock/GibbsQuadraticPenalty.h"


START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
CudaGibbsQuadraticPenalty<elemT>::CudaGibbsQuadraticPenalty()
{
  set_defaults();
}

template <typename elemT>
CudaGibbsQuadraticPenalty<elemT>::CudaGibbsQuadraticPenalty(const bool only_2D, float penalisation_factor)
  : base_type(only_2D, penalisation_factor) {}

template <typename elemT>
void
CudaGibbsQuadraticPenalty<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
std::string 
CudaGibbsQuadraticPenalty<elemT>::get_parsing_name() const
{
  return registered_name;
}

template class CudaGibbsQuadraticPenalty<float>;

END_NAMESPACE_STIR
