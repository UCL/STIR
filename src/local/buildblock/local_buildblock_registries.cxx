/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/DAVImageFilter3D.h"
#include "local/stir/ModifiedInverseAverigingImageFilter.h"
#include "local/stir/ModifiedInverseAveragingImageFilterAll.h"
#include "local/stir/SeparableLowPassImageFilter.h"
#include "local/stir/SeparableGaussianImageFilter.h"
//#include "local/stir/SeparableGaussianImageFilterWithSquareCoefficients.h"


START_NAMESPACE_STIR
static ModifiedInverseAverigingImageFilter<float>::RegisterIt dummy2;
static DAVImageFilter3D<float>::RegisterIt dummy1;

//static InverseSeparableCartesianMetzImageFilter <float>::RegisterIt dummy3;
static SeparableLowPassImageFilter<float>::RegisterIt dummy4;
static SeparableGaussianImageFilter<float>::RegisterIt dummy5;

static ModifiedInverseAveragingImageFilterAll<float>::RegisterIt dummy6;

END_NAMESPACE_STIR

