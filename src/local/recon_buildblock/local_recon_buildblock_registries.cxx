//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief File that registers all RegisterObject children in recon_buildblock

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifdef HAVE_LLN_MATRIX
#ifndef _MSC_VER 
// can't use it yet for VC because of conflict between matrix.h's ntohs and winsock2.h's ntohs
// TODO
#include "local/stir/recon_buildblock/BinNormalisationFromECAT7.h"
#endif
#endif

#include "local/stir/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#include "local/stir/recon_buildblock/QuadraticPrior.h"
//#include "local/stir/recon_buildblock/oldForwardProjectorByBinUsingRayTracing.h"
//#include "local/stir/recon_buildblock/oldBackProjectorByBinUsingInterpolation.h"
#include "local/stir/recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#include "local/stir/recon_buildblock/BinNormalisationUsingProfile.h"

START_NAMESPACE_STIR

static ProjMatrixByBinUsingSolidAngle::RegisterIt dummy11;

//static oldForwardProjectorByBinUsingRayTracing::RegisterIt dummy1;
static PostsmoothingForwardProjectorByBin::RegisterIt dummy2;
static PresmoothingForwardProjectorByBin::RegisterIt dummy3;
static PostsmoothingBackProjectorByBin::RegisterIt dummy4;
//static oldBackProjectorByBinUsingInterpolation::RegisterIt dummy5;

static QuadraticPrior<float>::RegisterIt dummy21;

static BinNormalisationUsingProfile::RegisterIt dummy101;
#ifdef HAVE_LLN_MATRIX
#ifndef _MSC_VER // TODO
START_NAMESPACE_ECAT7
static BinNormalisationFromECAT7::RegisterIt dummy102;
END_NAMESPACE_ECAT7
#endif
#endif

END_NAMESPACE_STIR
