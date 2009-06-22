//
// $Id$
//
/*!
  \file
  \ingroup priors
  \brief Inline implementations for class stir::GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

