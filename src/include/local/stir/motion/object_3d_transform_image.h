//
// $Id$
//
/*!

  \file
  \ingroup motion
  \brief Declaration of functions to re-interpolate an image to a new coordinate system.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir/common.h"

START_NAMESPACE_STIR

class RigidObject3DTransformation;
template <int num_dimensions, typename elemT> 
  class DiscretisedDensity;


//! transform image data
/*! Currently only supports VoxelOnCartesianGrid image.
    Uses nearest neighbourhood interpolation.
*/

void 
object_3d_transform_image(DiscretisedDensity<3,float>& out_density, 
			  const DiscretisedDensity<3,float>& in_density, 
			  const RigidObject3DTransformation& rigid_object_transformation);


END_NAMESPACE_STIR
