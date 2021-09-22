//
//
/*
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup Shape

  \brief File that registers all stir::RegisteredObject children in Shape_buildblock

  \author Kris Thielemans
  \author C. Ross Schmidtlein (added stir::Box3D Shape class)
  
*/

#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"
#include "stir/Shape/DiscretisedShape3D.h"

START_NAMESPACE_STIR

static Ellipsoid::RegisterIt dummy;
static EllipsoidalCylinder::RegisterIt dummy2;
static DiscretisedShape3D::RegisterIt dummy3;
static Box3D::RegisterIt dummy4;

END_NAMESPACE_STIR
