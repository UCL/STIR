//
//
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief  implementation of the stir::GeneralisedPrior
    
  \author Kris Thielemans
  \author Sanida Mustafovic      
*/

#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

START_NAMESPACE_STIR


template <typename TargetT>
void 
GeneralisedPrior<TargetT>::initialise_keymap()
{
  this->parser.add_key("penalisation factor", &this->penalisation_factor); 
}


template <typename TargetT>
void
GeneralisedPrior<TargetT>::set_defaults()
{
  _already_set_up = false;
  this->penalisation_factor = 0;  
}

template <typename TargetT>
Succeeded 
GeneralisedPrior<TargetT>::
set_up(shared_ptr<const TargetT> const&)
{
  _already_set_up = true;
  return Succeeded::yes;
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::
compute_Hessian(TargetT& output,
                const BasicCoordinate<3,int>& coords,
                const TargetT& current_image_estimate) const
{
  if (this->is_convex())
    error("GeneralisedPrior:\n  compute_Hessian implementation is not overloaded by your convex prior.");
  else
    error("GeneralisedPrior:\n  compute_Hessian is not implemented for this (non-convex) prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::
add_multiplication_with_approximate_Hessian(TargetT& output,
              const TargetT& input) const
{
  error("GeneralisedPrior:\n"
  "add_multiplication_with_approximate_Hessian implementation is not overloaded by your prior.");
}

template <typename TargetT>
void
GeneralisedPrior<TargetT>::
accumulate_Hessian_times_input(TargetT& output,
        const TargetT& current_estimate,
        const TargetT& input) const
{
  error("GeneralisedPrior:\n"
        "accumulate_Hessian_times_input implementation is not overloaded by your prior.");
}

template <typename TargetT> 
void GeneralisedPrior<TargetT>::check(TargetT const& current_estimate) const
{
  if (!_already_set_up) 
    error("The prior should already be set-up, but it's not.");
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class GeneralisedPrior<DiscretisedDensity<3,float> >;
template class GeneralisedPrior<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR
