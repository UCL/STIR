//
//
/*!
  \file
  \ingroup Shape

  \brief Inline-implementations of class stir::DiscretisedShape3D

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR


const VoxelsOnCartesianGrid<float>& 
DiscretisedShape3D::
image() const
{
  return static_cast<const VoxelsOnCartesianGrid<float>&>(*density_ptr);
}
 
VoxelsOnCartesianGrid<float>& 
DiscretisedShape3D::
image()
{
  return static_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);
}

END_NAMESPACE_STIR
