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

#include "tomo/recon_buildblock/FilterRootPrior.h"

#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include "recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"

#include "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/BackProjectorByBinUsingInterpolation.h"

#include "recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#include "tomo/recon_buildblock/TrivialBinNormalisation.h"
#include "tomo/recon_buildblock/BinNormalisationFromProjData.h"

START_NAMESPACE_TOMO

static FilterRootPrior<float>::RegisterIt dummy4;

static ProjMatrixByBinUsingRayTracing::RegisterIt dummy11;

static ForwardProjectorByBinUsingProjMatrixByBin::RegisterIt dummy31;
static ForwardProjectorByBinUsingRayTracing::RegisterIt dummy32;

static BackProjectorByBinUsingProjMatrixByBin::RegisterIt dummy51;
static BackProjectorByBinUsingInterpolation::RegisterIt dummy52;

static ProjectorByBinPairUsingProjMatrixByBin::RegisterIt dummy71;
static ProjectorByBinPairUsingSeparateProjectors::RegisterIt dummy72;

static TrivialBinNormalisation::RegisterIt dummy91;
static BinNormalisationFromProjData::RegisterIt dummy92;

END_NAMESPACE_TOMO
