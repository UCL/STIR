//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief This file defines two private static functions from
  BackProjectorByBinUsingInterpolation, for the case of piecewise 
  linear interpolation.

  \warning This #includes BackProjectorByBinUsingInterpolation_3DCho.cxx 

  This very ugly way of including a .cxx file is used to avoid replication of
  a lot of (difficult) code.

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#define PIECEWISE_INTERPOLATION 0
#include "BackProjectorByBinUsingInterpolation_3DCho.cxx"
#undef PIECEWISE_INTERPOLATION
