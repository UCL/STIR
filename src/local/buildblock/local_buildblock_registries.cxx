/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/DAVImageFilter3D.h"
#include "local/stir/ModifiedInverseAverigingImageFilter.h"
#include "local/stir/ModifiedInverseAveragingImageFilterAll.h"
#include "local/stir/SeparableLowPassImageFilter.h"
#include "local/stir/SeparableGaussianImageFilter.h"
#include "local/stir/NonseparableSpatiallyVaryingFilters.h"
#include "local/stir/NonseparableSpatiallyVaryingFilters3D.h"
#include "local/stir/cleanup966ImageProcessor.h"
#include "local/stir/multiply_plane_scale_factorsImageProcessor.h"

START_NAMESPACE_STIR
static ModifiedInverseAverigingImageFilter<float>::RegisterIt dummy2;
static DAVImageFilter3D<float>::RegisterIt dummy1;

//static InverseSeparableCartesianMetzImageFilter <float>::RegisterIt dummy3;
static SeparableLowPassImageFilter<float>::RegisterIt dummy4;
static ModifiedInverseAveragingImageFilterAll<float>::RegisterIt dummy6;
static NonseparableSpatiallyVaryingFilters<float>:: RegisterIt dummy7;
static NonseparableSpatiallyVaryingFilters3D<float>::RegisterIt dummy8;

static cleanup966ImageProcessor<float>::RegisterIt dummy100;
static multiply_plane_scale_factorsImageProcessor<float>::RegisterIt dummy101;

END_NAMESPACE_STIR

