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
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/SeparableCartesianMetzImageFilter.h"
#include "stir/MedianImageFilter3D.h"
#include "stir/ChainedImageProcessor.h"
#include "stir/TruncateMinToSmallPositiveValueImageProcessor.h"

START_NAMESPACE_STIR

static MedianImageFilter3D<float>::RegisterIt dummy;
static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy2;
static ChainedImageProcessor<3,float>::RegisterIt dummy3;
static TruncateMinToSmallPositiveValueImageProcessor<float>::RegisterIt dummy4;
END_NAMESPACE_STIR
