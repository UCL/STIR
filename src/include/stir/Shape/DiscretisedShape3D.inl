//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Inline-implementations of class DiscretisedShape3D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
