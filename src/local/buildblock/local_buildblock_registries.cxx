#include "local/stir/DAVImageFilter3D.h"
#include "local/stir/ModifiedInverseAverigingImageFilter.h"
//#include "local/stir/InverseSeparableCartesianMetzImageFilter.h"
#include "local/stir/SeparableLowPassImageFilter.h"


START_NAMESPACE_STIR
static ModifiedInverseAverigingImageFilter<float>::RegisterIt dummy2;
static DAVImageFilter3D<float>::RegisterIt dummy1;

//static InverseSeparableCartesianMetzImageFilter <float>::RegisterIt dummy3;
static SeparableLowPassImageFilter<float>::RegisterIt dummy4;

END_NAMESPACE_STIR
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
