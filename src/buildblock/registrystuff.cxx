#include "tomo/SeparableCartesianMetzImageFilter.h"
#include "tomo/MedianImageFilter3D.h"
//#include "tomo/DAVImageFilter3D.h"
#include "tomo/recon_buildblock/FilterRootPrior.h"

START_NAMESPACE_TOMO

static MedianImageFilter3D<float>::RegisterIt dummy;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
//static DAVImageFilter3D<float>::RegisterIt dummy3;

static FilterRootPrior<float>::RegisterIt dummy4;
END_NAMESPACE_TOMO
