//
//
/*!
  \file
  \ingroup priors
  \brief Inline implementations for class stir::GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR


template <typename elemT>
GeneralisedPrior<elemT>::GeneralisedPrior() 
{ 
  penalisation_factor =0;
}


template <typename elemT>
float
GeneralisedPrior<elemT>::
get_penalisation_factor() const
{ return penalisation_factor; }

template <typename elemT>
void
GeneralisedPrior<elemT>::
set_penalisation_factor(const float new_penalisation_factor)
{ penalisation_factor = new_penalisation_factor; }

END_NAMESPACE_STIR

