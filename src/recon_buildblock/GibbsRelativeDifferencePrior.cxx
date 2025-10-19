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
  \brief Implementation of class stir::GibbsRelativeDifferencePrior

  \author Matteo Neel Colombo
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/GibbsRelativeDifferencePrior.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

// Implementation of constructors
template <typename elemT>
GibbsRelativeDifferencePrior<elemT>::GibbsRelativeDifferencePrior()
{
  set_defaults();
}

template <typename elemT>
GibbsRelativeDifferencePrior<elemT>::GibbsRelativeDifferencePrior(const bool only_2D,
                                                                  float penalisation_factor,
                                                                  float gamma_v,
                                                                  float epsilon_v)
    : base_type(only_2D, penalisation_factor)
{
  this->potential.gamma = gamma_v;
  this->potential.epsilon = epsilon_v;
}

template <typename elemT>
void
GibbsRelativeDifferencePrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->potential.gamma = 2;
  this->potential.epsilon = 1e-7F;
}

template <typename elemT>
std::string 
GibbsRelativeDifferencePrior<elemT>::get_parsing_name() const
{
  return registered_name;
}

template class GibbsRelativeDifferencePrior<float>;

END_NAMESPACE_STIR
