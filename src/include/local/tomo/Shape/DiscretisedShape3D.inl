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
#include "VoxelsOnCartesianGrid.h"

START_NAMESPACE_TOMO


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

END_NAMESPACE_TOMO
