//
// $Id$
//
/*!
  \file
  \ingroup densitydata

  \brief Declaration of typedef stir::Densel
  \author Kris Thielemans

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
#ifndef __stir_Densel__H_
#define __stir_Densel__H_
//TODO
#include "stir/Coordinate3D.h"

START_NAMESPACE_STIR

/*! \ingroup densitydata
\brief a typedef used for an element of a DiscretisedDensity

The name is a generalisation of pixel/voxel.

\todo This might at some point evolve into a class, similar to Bin. 
\warning At the moment,
Bin includes a value, while Densel does not.

*/
typedef Coordinate3D<int> Densel;

END_NAMESPACE_STIR

#endif
