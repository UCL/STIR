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
GibbsRelativeDifferencePrior<elemT>::GibbsRelativeDifferencePrior(const bool only_2D, float penalisation_factor, float gamma_v, float epsilon_v)
  : base_type(only_2D, penalisation_factor)  {this->potential.gamma = gamma_v; this->potential.epsilon = epsilon_v;}

template <typename elemT>
void
GibbsRelativeDifferencePrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->potential.gamma = 2;
  this->potential.epsilon = 1e-7F;
}

template <typename elemT>
void
GibbsRelativeDifferencePrior<elemT>::initialise_keymap()
{
  this->parser.add_start_key("Gibbs Relative Difference Prior Parameters");
  base_type::initialise_keymap();
  this->parser.add_key("gamma value", &this->potential.gamma);
  this->parser.add_key("epsilon value", &this->potential.epsilon);
  this->parser.add_stop_key("END Gibbs Relative Difference Prior Parameters");
}


template class GibbsRelativeDifferencePrior<float>;

END_NAMESPACE_STIR
