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


#include "local/stir/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#include "local/stir/recon_buildblock/ProjMatrixByBinUsingInterpolation.h"
#include "local/stir/recon_buildblock/ProjMatrixByBinSinglePhoton.h"
#include "local/stir/recon_buildblock/ProjMatrixByBinFromFile.h"

#include "local/stir/recon_buildblock/QuadraticPrior.h"
//#include "local/stir/recon_buildblock/NonquadraticPriorWithNaturalLogarithm.h"
//#include "local/stir/recon_buildblock/oldForwardProjectorByBinUsingRayTracing.h"
//#include "local/stir/recon_buildblock/oldBackProjectorByBinUsingInterpolation.h"
#include "local/stir/recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#include "local/stir/recon_buildblock/BinNormalisationUsingProfile.h"
#include "local/stir/recon_buildblock/BinNormalisationSinogramRescaling.h"


START_NAMESPACE_STIR

static ProjMatrixByBinUsingSolidAngle::RegisterIt dummy11;
static ProjMatrixByBinUsingInterpolation::RegisterIt dummy13;
static ProjMatrixByBinSinglePhoton::RegisterIt dummy12;
static ProjMatrixByBinFromFile::RegisterIt dumy14;
//static NonquadraticPriorWithNaturalLogarithm<float>::RegisterIt dummy22;

//static oldForwardProjectorByBinUsingRayTracing::RegisterIt dummy1;
static PostsmoothingForwardProjectorByBin::RegisterIt dummy2;
static PresmoothingForwardProjectorByBin::RegisterIt dummy3;
static PostsmoothingBackProjectorByBin::RegisterIt dummy4;
//static oldBackProjectorByBinUsingInterpolation::RegisterIt dummy5;

static QuadraticPrior<float>::RegisterIt dummy21;


static BinNormalisationUsingProfile::RegisterIt dummy101;


static BinNormalisationSinogramRescaling::RegisterIt dummy555;


END_NAMESPACE_STIR
