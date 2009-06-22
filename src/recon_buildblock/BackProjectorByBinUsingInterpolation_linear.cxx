//
// $Id$
//
/*!

  \file
  \ingroup projection

  \brief This file defines two private static functions from
  stir::BackProjectorByBinUsingInterpolation, for the case of piecewise 
  linear interpolation.

  \warning This #includes BackProjectorByBinUsingInterpolation_3DCho.cxx 

  This very ugly way of including a .cxx file is used to avoid replication of
  a lot of (difficult) code.

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
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

#define PIECEWISE_INTERPOLATION 0
#include "BackProjectorByBinUsingInterpolation_3DCho.cxx"
#undef PIECEWISE_INTERPOLATION
