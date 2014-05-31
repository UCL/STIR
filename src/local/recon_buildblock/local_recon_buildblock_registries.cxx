//
//
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    For GE internal use only.
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief File that registers all RegisterObject children in recon_buildblock

  \author Kris Thielemans
*/


#if 0
#include "local/stir/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#include "local/stir/recon_buildblock/ProjMatrixByBinSinglePhoton.h"
#endif

//#include "local/stir/recon_buildblock/BackProjectorByBinDistanceDriven.h"
//#include "local/stir/recon_buildblock/ForwardProjectorByBinDistanceDriven.h"

//#include "local/stir/recon_buildblock/NonquadraticPriorWithNaturalLogarithm.h"
//#include "local/stir/recon_buildblock/oldForwardProjectorByBinUsingRayTracing.h"
//#include "local/stir/recon_buildblock/oldBackProjectorByBinUsingInterpolation.h"
#include "local/stir/recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "local/stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#if 0
#include "local/stir/recon_buildblock/BinNormalisationUsingProfile.h"
#include "local/stir/recon_buildblock/BinNormalisationSinogramRescaling.h"
//#include "stir/recon_buildblock/FilterRootPrior.h"
#include "local/stir/recon_buildblock/ParametricQuadraticPrior.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#endif

START_NAMESPACE_STIR

#if 0
static ProjMatrixByBinUsingSolidAngle::RegisterIt dummy11;
static ProjMatrixByBinSinglePhoton::RegisterIt dummy12;
static PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid>::RegisterIt Dummyxxx;

//static FilterRootPrior<ParametricVoxelsOnCartesianGrid>::RegisterIt dummy44;
static ParametricQuadraticPrior<ParametricVoxelsOnCartesianGrid>::RegisterIt dummy5;
#endif

//static PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<DynamicDiscretisedDensity>::RegisterIt Dummyyyy;
//static BackProjectorByBinDistanceDriven::RegisterIt dummy1001;
//static ForwardProjectorByBinDistanceDriven::RegisterIt dummy1002;

//static NonquadraticPriorWithNaturalLogarithm<float>::RegisterIt dummy22;

//static oldForwardProjectorByBinUsingRayTracing::RegisterIt dummy1;
static PostsmoothingForwardProjectorByBin::RegisterIt dummy2;
static PresmoothingForwardProjectorByBin::RegisterIt dummy3;
static PostsmoothingBackProjectorByBin::RegisterIt dummy4;
//static oldBackProjectorByBinUsingInterpolation::RegisterIt dummy5;

#if 0

static BinNormalisationUsingProfile::RegisterIt dummy101;


static BinNormalisationSinogramRescaling::RegisterIt dummy555;
#endif

END_NAMESPACE_STIR
