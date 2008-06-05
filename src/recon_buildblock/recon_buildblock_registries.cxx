//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute that part and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief File that registers all stir::RegisterObject children in recon_buildblock

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h"

#include "stir/recon_buildblock/FilterRootPrior.h"
#include "stir/DataProcessor.h"
#include "stir/recon_buildblock/QuadraticPrior.h"

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"

#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"

#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"

#ifdef HAVE_LLN_MATRIX
#include "stir/recon_buildblock/BinNormalisationFromECAT7.h"
#endif

START_NAMESPACE_STIR

static PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float> >::RegisterIt dummy1;
static PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3,float> >::RegisterIt dummy2;

static FilterRootPrior<DiscretisedDensity<3,float> >::RegisterIt dummy4;
static QuadraticPrior<float>::RegisterIt dummy5;

static ProjMatrixByBinUsingRayTracing::RegisterIt dummy11;

static ForwardProjectorByBinUsingProjMatrixByBin::RegisterIt dummy31;
static ForwardProjectorByBinUsingRayTracing::RegisterIt dummy32;

static BackProjectorByBinUsingProjMatrixByBin::RegisterIt dummy51;
static BackProjectorByBinUsingInterpolation::RegisterIt dummy52;

static ProjectorByBinPairUsingProjMatrixByBin::RegisterIt dummy71;
static ProjectorByBinPairUsingSeparateProjectors::RegisterIt dummy72;

static TrivialBinNormalisation::RegisterIt dummy91;
static ChainedBinNormalisation::RegisterIt dummy92;
static BinNormalisationFromProjData::RegisterIt dummy93;
static BinNormalisationFromAttenuationImage::RegisterIt dummy94;
#ifdef HAVE_LLN_MATRIX
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
static BinNormalisationFromECAT7::RegisterIt dummy102;
END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
#endif

END_NAMESPACE_STIR
