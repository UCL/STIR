//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2019, 2020, 2024 UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock

  \brief File that registers all stir::RegisterObject children in buildblock

  \author Kris Thielemans
  \author Dimitra Kyriakopoulou
*/

#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/MedianImageFilter3D.h"
#include "stir/WienerImageFilter2D.h"
#include "stir/GammaImageFilter2D.h"
#include "stir/MinimalImageFilter3D.h"
#include "stir/ChainedDataProcessor.h"
#include "stir/ThresholdMinToSmallPositiveValueDataProcessor.h"
#include "stir/SeparableConvolutionImageFilter.h"
#include "stir/NonseparableConvolutionUsingRealDFTImageFilter.h"
#include "stir/TruncateToCylindricalFOVImageProcessor.h"
#ifdef HAVE_JSON
#  include "stir/HUToMuImageProcessor.h"
#endif
START_NAMESPACE_STIR

static MedianImageFilter3D<float>::RegisterIt dummy;
static WienerImageFilter2D<float>::RegisterIt dummyWiener;
static GammaImageFilter2D<float>::RegisterIt dummyGamma;
static MinimalImageFilter3D<float>::RegisterIt dummy1;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
static SeparableGaussianImageFilter<float>::RegisterIt dummySGF;
static SeparableConvolutionImageFilter<float>::RegisterIt dummy5;
static NonseparableConvolutionUsingRealDFTImageFilter<float>::RegisterIt dummy7;
static TruncateToCylindricalFOVImageProcessor<float>::RegisterIt dummy6;
static ChainedDataProcessor<DiscretisedDensity<3, float>>::RegisterIt dummy3;
static ThresholdMinToSmallPositiveValueDataProcessor<DiscretisedDensity<3, float>>::RegisterIt dummy4;

#ifdef HAVE_JSON
static HUToMuImageProcessor<DiscretisedDensity<3, float>>::RegisterIt dummyHUToMu;
#endif
END_NAMESPACE_STIR
