
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

END_NAMESPACE_STIR
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
