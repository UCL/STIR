//
//
/*
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
