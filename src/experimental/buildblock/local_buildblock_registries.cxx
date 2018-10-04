/*
    Copyright (C) 2000- 2010, IRSL
    See STIR/LICENSE.txt for detail $, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir_experimental/SeparableGaussianImageFilter.h"
#ifdef SANIDA
#include "stir_experimental/DAVImageFilter3D.h"
#include "stir_experimental/ModifiedInverseAverigingImageFilter.h"
#include "stir_experimental/ModifiedInverseAveragingImageFilterAll.h"
#include "stir_experimental/NonseparableSpatiallyVaryingFilters.h"
#include "stir_experimental/NonseparableSpatiallyVaryingFilters3D.h"
#include "stir_experimental/multiply_plane_scale_factorsImageProcessor.h"
#endif

#if 0
#include "stir_experimental/AbsTimeIntervalWithParsing.h"
#ifdef HAVE_LLN_MATRIX
#include "stir_experimental/AbsTimeIntervalFromECAT7ACF.h"
#endif
#include "stir_experimental/AbsTimeIntervalFromDynamicData.h"
#endif

START_NAMESPACE_STIR
static SeparableGaussianImageFilter<float>::RegisterIt dummy4;
#ifdef SANIDA
static ModifiedInverseAverigingImageFilter<float>::RegisterIt dummy2;
static DAVImageFilter3D<float>::RegisterIt dummy1;

//static InverseSeparableCartesianMetzImageFilter <float>::RegisterIt dummy3;
static SeparableLowPassImageFilter<float>::RegisterIt dummy4;
//static ModifiedInverseAveragingImageFilterAll<float>::RegisterIt dummy6;
static NonseparableSpatiallyVaryingFilters<float>:: RegisterIt dummy7;
static NonseparableSpatiallyVaryingFilters3D<float>::RegisterIt dummy8;
#endif

#if 0
static multiply_plane_scale_factorsImageProcessor<float>::RegisterIt dummy101;

static AbsTimeIntervalWithParsing::RegisterIt dummy200;
#ifdef HAVE_LLN_MATRIX
static AbsTimeIntervalFromECAT7ACF::RegisterIt dummy201;
#endif
static AbsTimeIntervalFromDynamicData::RegisterIt dummy202;
#endif

END_NAMESPACE_STIR

