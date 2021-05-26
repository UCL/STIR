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
#  include "stir_experimental/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#  include "stir_experimental/recon_buildblock/ProjMatrixByBinSinglePhoton.h"
#endif

//#include "stir_experimental/recon_buildblock/BackProjectorByBinDistanceDriven.h"
//#include "stir_experimental/recon_buildblock/ForwardProjectorByBinDistanceDriven.h"

//#include "stir_experimental/recon_buildblock/NonquadraticPriorWithNaturalLogarithm.h"
//#include "stir_experimental/recon_buildblock/oldForwardProjectorByBinUsingRayTracing.h"
//#include "stir_experimental/recon_buildblock/oldBackProjectorByBinUsingInterpolation.h"
#include "stir_experimental/recon_buildblock/PostsmoothingForwardProjectorByBin.h"
#if 0
#  include "stir_experimental/recon_buildblock/BinNormalisationUsingProfile.h"
#  include "stir_experimental/recon_buildblock/BinNormalisationSinogramRescaling.h"
//#include "stir/recon_buildblock/FilterRootPrior.h"
#  include "stir_experimental/recon_buildblock/ParametricQuadraticPrior.h"
#  include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.h"
#  include "stir/modelling/ParametricDiscretisedDensity.h"
#  include "stir/DynamicDiscretisedDensity.h"
#endif

START_NAMESPACE_STIR

#if 0
static ProjMatrixByBinUsingSolidAngle::RegisterIt dummy11;
static ProjMatrixByBinSinglePhoton::RegisterIt dummy12;
static PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid>::RegisterIt Dummyxxx;

//static FilterRootPrior<ParametricVoxelsOnCartesianGrid>::RegisterIt dummy44;
static ParametricQuadraticPrior<ParametricVoxelsOnCartesianGrid>::RegisterIt dummy5;
#endif

// static PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<DynamicDiscretisedDensity>::RegisterIt Dummyyyy;
// static BackProjectorByBinDistanceDriven::RegisterIt dummy1001;
// static ForwardProjectorByBinDistanceDriven::RegisterIt dummy1002;

// static NonquadraticPriorWithNaturalLogarithm<float>::RegisterIt dummy22;

// static oldForwardProjectorByBinUsingRayTracing::RegisterIt dummy1;
static PostsmoothingForwardProjectorByBin::RegisterIt dummy2;
// static oldBackProjectorByBinUsingInterpolation::RegisterIt dummy5;

#if 0

static BinNormalisationUsingProfile::RegisterIt dummy101;


static BinNormalisationSinogramRescaling::RegisterIt dummy555;
#endif

END_NAMESPACE_STIR
