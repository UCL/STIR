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

#include "local/tomo/Shape/Ellipsoid.h"
#include "local/tomo/Shape/EllipsoidalCylinder.h"
#include "local/tomo/Shape/DiscretisedShape3D.h"

START_NAMESPACE_TOMO

static Ellipsoid::RegisterIt dummy;
static EllipsoidalCylinder::RegisterIt dummy2;
static DiscretisedShape3D::RegisterIt dummy3;

END_NAMESPACE_TOMO
