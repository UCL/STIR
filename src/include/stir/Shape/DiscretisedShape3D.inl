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

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR


const VoxelsOnCartesianGrid<float>& 
DiscretisedShape3D::
image() const
{
  return static_cast<const VoxelsOnCartesianGrid<float>&>(*density_sptr);
}
 
#if 0
VoxelsOnCartesianGrid<float>& 
DiscretisedShape3D::
image()
{
  return static_cast<const VoxelsOnCartesianGrid<float>&>(*density_sptr);
}
#endif
END_NAMESPACE_STIR
