#include "tomo/SeparableCartesianMetzImageFilter.h"
#include "tomo/MedianImageFilter3D.h"

START_NAMESPACE_TOMO

static MedianImageFilter3D<float>::RegisterIt dummy;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
END_NAMESPACE_TOMO
