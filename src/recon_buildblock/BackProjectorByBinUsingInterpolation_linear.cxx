//
// $Id$
//
/*!

  \file
  \ingroup projection

  \brief This file defines two private static functions from
  BackProjectorByBinUsingInterpolation, for the case of piecewise 
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
    See STIR/LICENSE.txt for details
*/

#define PIECEWISE_INTERPOLATION 0
#include "BackProjectorByBinUsingInterpolation_3DCho.cxx"
#undef PIECEWISE_INTERPOLATION
