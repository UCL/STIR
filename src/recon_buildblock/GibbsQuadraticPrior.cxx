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
  \brief Implementation of class stir::GibbsQuadraticPrior

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsQuadraticPrior.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
GibbsQuadraticPrior<elemT>::GibbsQuadraticPrior()
{
  set_defaults();
}

template <typename elemT>
GibbsQuadraticPrior<elemT>::GibbsQuadraticPrior(const bool only_2D, float penalisation_factor)
    : base_type(only_2D, penalisation_factor)
{}

template <typename elemT>
void
GibbsQuadraticPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
}

template <typename elemT>
std::string
GibbsQuadraticPrior<elemT>::get_parsing_name() const
{
  return registered_name;
}

// Explicit template instantiations
template class GibbsQuadraticPrior<float>;

END_NAMESPACE_STIR