//
//
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
*/
#include "stir/DiscretisedDensity.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h"

#include "stir/recon_buildblock/FilterRootPrior.h"
#include "stir/DataProcessor.h"
#include "stir/recon_buildblock/QuadraticPrior.h"

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingInterpolation.h"
#include "stir/recon_buildblock/ProjMatrixByBinFromFile.h"
#include "stir/recon_buildblock/ProjMatrixByBinSPECTUB.h"

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"

#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"

#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.h"

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"

#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/OSSPS/OSSPSReconstruction.h"

#ifdef HAVE_LLN_MATRIX
#include "stir/recon_buildblock/BinNormalisationFromECAT7.h"
#endif
#include "stir/recon_buildblock/BinNormalisationFromECAT8.h"

#ifdef HAVE_HDF5
#include "stir/recon_buildblock/BinNormalisationFromGEHDF5.h"
#endif

#include "stir/recon_buildblock/FourierRebinning.h"

//#include "stir/IO/InputFileFormatRegistry.h"

START_NAMESPACE_STIR
//static RegisterInputFileFormat<InterfileProjMatrixByBinInputFileFormat> idummy0(0);

static PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float> >::RegisterIt dummy1;
static PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3,float> >::RegisterIt dummy2;

static FilterRootPrior<DiscretisedDensity<3,float> >::RegisterIt dummy4;
static QuadraticPrior<float>::RegisterIt dummy5;

static ProjMatrixByBinUsingRayTracing::RegisterIt dummy11;
static ProjMatrixByBinUsingInterpolation::RegisterIt dummy12;
static ProjMatrixByBinFromFile::RegisterIt dumy13;
static ProjMatrixByBinSPECTUB::RegisterIt dumy14;

static ForwardProjectorByBinUsingProjMatrixByBin::RegisterIt dummy31;
static ForwardProjectorByBinUsingRayTracing::RegisterIt dummy32;
static PostsmoothingBackProjectorByBin::RegisterIt dummy33;

static BackProjectorByBinUsingProjMatrixByBin::RegisterIt dummy51;
static BackProjectorByBinUsingInterpolation::RegisterIt dummy52;
static PresmoothingForwardProjectorByBin::RegisterIt dummy53;

static ProjectorByBinPairUsingProjMatrixByBin::RegisterIt dummy71;
static ProjectorByBinPairUsingSeparateProjectors::RegisterIt dummy72;

static TrivialBinNormalisation::RegisterIt dummy91;
static ChainedBinNormalisation::RegisterIt dummy92;
static BinNormalisationFromProjData::RegisterIt dummy93;
static BinNormalisationFromAttenuationImage::RegisterIt dummy94;
static PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<ParametricVoxelsOnCartesianGrid>::RegisterIt Dummyxxx;
static PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<DiscretisedDensity<3,float> >::RegisterIt Dummyxxxzz;

static FBP2DReconstruction::RegisterIt dummy601;
static FBP3DRPReconstruction::RegisterIt dummy602;

static OSMAPOSLReconstruction<DiscretisedDensity<3,float> >::RegisterIt dummy603;
static OSSPSReconstruction<DiscretisedDensity<3, float> >::RegisterIt dummy604;

#ifdef HAVE_LLN_MATRIX
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
static BinNormalisationFromECAT7::RegisterIt dummy102;
END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
#endif

START_NAMESPACE_ECAT
static BinNormalisationFromECAT8::RegisterIt dummy103;
END_NAMESPACE_ECAT

#ifdef HAVE_HDF5
START_NAMESPACE_ECAT
static BinNormalisationFromGEHDF5::RegisterIt dummy104;
END_NAMESPACE_ECAT
#endif

static FourierRebinning::RegisterIt dummyFORE;

END_NAMESPACE_STIR
