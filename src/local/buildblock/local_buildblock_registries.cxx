#include "local/tomo/DAVImageFilter3D.h"
#include "local/tomo/ModifiedInverseAverigingImageFilter.h"
//#include "local/tomo/InverseSeparableCartesianMetzImageFilter.h"
#include "local/tomo/SeparableLowPassImageFilter.h"


START_NAMESPACE_TOMO
static ModifiedInverseAverigingImageFilter<float>::RegisterIt dummy2;
static DAVImageFilter3D<float>::RegisterIt dummy1;

//static InverseSeparableCartesianMetzImageFilter <float>::RegisterIt dummy3;
static SeparableLowPassImageFilter<float>::RegisterIt dummy4;

END_NAMESPACE_TOMO
