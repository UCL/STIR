//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*
  \file 
  \ingroup buildblock
  \brief Generic Function to sample 3D arrays

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

/*!
\brief Generic Function to sample 3D arrays
\ingroup buildblock
\brief Extension of the 2D sinograms in view direction.
	Its propose is to pick up sample values at given points from an array. 
	At the moment, only a simple function is implemented to have as output a 3D regular array.
*/
template <class Function3D, class elemT, class positionT>
inline
void sample_at_regular_array(Array<3,elemT>& out,
				 Function3D func,
				 const BasicCoordinate<3, positionT>&  offset,  
				 const BasicCoordinate<3, positionT>& step);

END_NAMESPACE_STIR

#include "local/stir/sample_array.inl"
