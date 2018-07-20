//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
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
  \ingroup buildblock

  \brief File that registers all stir::RegisterObject children in buildblock

  \author Kris Thielemans
  
*/

#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/MedianImageFilter3D.h"
#include "stir/MinimalImageFilter3D.h"
#include "stir/ChainedDataProcessor.h"
#include "stir/ThresholdMinToSmallPositiveValueDataProcessor.h"
#include "stir/SeparableConvolutionImageFilter.h"
#include "stir/NonseparableConvolutionUsingRealDFTImageFilter.h"
#include "stir/TruncateToCylindricalFOVImageProcessor.h"
START_NAMESPACE_STIR

static MedianImageFilter3D<float>::RegisterIt dummy;
static MinimalImageFilter3D<float>::RegisterIt dummy1;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
static SeparableGaussianImageFilter<float>::RegisterIt dummy9;
static SeparableConvolutionImageFilter<float>::RegisterIt dummy5;
static NonseparableConvolutionUsingRealDFTImageFilter<float>::RegisterIt dummy7;
static TruncateToCylindricalFOVImageProcessor<float> ::RegisterIt dummy6;
static ChainedDataProcessor<DiscretisedDensity<3,float> >::RegisterIt dummy3;
static ThresholdMinToSmallPositiveValueDataProcessor<DiscretisedDensity<3,float> >::RegisterIt dummy4;
END_NAMESPACE_STIR
