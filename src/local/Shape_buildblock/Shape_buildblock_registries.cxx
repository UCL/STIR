//
// $Id$
//
/*!

  \file
  \ingroup Shape

  \brief File that registers all RegisterObject children in Shape_buildblock

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/Shape/Ellipsoid.h"
#include "local/stir/Shape/EllipsoidalCylinder.h"
#include "local/stir/Shape/DiscretisedShape3D.h"

START_NAMESPACE_STIR

static Ellipsoid::RegisterIt dummy;
static EllipsoidalCylinder::RegisterIt dummy2;
static DiscretisedShape3D::RegisterIt dummy3;

END_NAMESPACE_STIR
