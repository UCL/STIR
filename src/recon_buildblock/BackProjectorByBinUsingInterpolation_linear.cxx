//
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


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#define PIECEWISE_INTERPOLATION 0
#include "BackProjectorByBinUsingInterpolation_3DCho.cxx"
#undef PIECEWISE_INTERPOLATION
