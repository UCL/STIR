//
//
/*!
  \file
  \ingroup densitydata

  \brief Declaration of typedef stir::Densel
  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
