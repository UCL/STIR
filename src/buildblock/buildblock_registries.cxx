//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief File that registers all RegisterObject children in buildblock

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "tomo/SeparableCartesianMetzImageFilter.h"
#include "tomo/MedianImageFilter3D.h"
#include "tomo/ChainedImageProcessor.h"
#include "tomo/TruncateMinToSmallPositiveValueImageProcessor.h"

START_NAMESPACE_TOMO

static MedianImageFilter3D<float>::RegisterIt dummy;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
static ChainedImageProcessor<3,float>::RegisterIt dummy3;
static TruncateMinToSmallPositiveValueImageProcessor<float>::RegisterIt dummy4;
END_NAMESPACE_TOMO
