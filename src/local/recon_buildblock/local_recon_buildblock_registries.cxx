
#include "local/tomo/recon_buildblock/ProjMatrixByBinUsingSolidAngle.h"
#include "local/tomo/recon_buildblock/QuadraticPrior.h"
#include "local/recon_buildblock/oldForwardProjectorByBinUsingRayTracing.h"
#include "local/recon_buildblock/oldBackProjectorByBinUsingInterpolation.h"
#include "local/recon_buildblock/PostsmoothingForwardProjectorByBin.h"

START_NAMESPACE_TOMO

static ProjMatrixByBinUsingSolidAngle::RegisterIt dummy11;

static oldForwardProjectorByBinUsingRayTracing::RegisterIt dummy1;
static PostsmoothingForwardProjectorByBin::RegisterIt dummy2;
static oldBackProjectorByBinUsingInterpolation::RegisterIt dummy3;

static QuadraticPrior<float>::RegisterIt dummy21;
END_NAMESPACE_TOMO
