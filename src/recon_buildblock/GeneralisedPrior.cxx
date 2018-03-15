//
//
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
set_up(shared_ptr<TargetT> const&)
{
  _already_set_up = true;
  return Succeeded::yes;
}

template <typename TargetT>
Succeeded 
GeneralisedPrior<TargetT>::
add_multiplication_with_approximate_Hessian(TargetT& output,
					    const TargetT& input) const
{
  error("GeneralisedPrior:\n"
	"add_multiplication_with_approximate_Hessian implementation is not overloaded by your prior.");
  return Succeeded::no;
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class GeneralisedPrior<DiscretisedDensity<3,float> >;
template class GeneralisedPrior<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR
