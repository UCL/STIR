//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For internal GE use only
*/
/*!

  \file
  \ingroup motion
  \brief Declaration of functions to re-interpolate an image or projection data 
  to a new coordinate system.

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/common.h"

START_NAMESPACE_STIR

class RigidObject3DTransformation;
template <int num_dimensions, typename elemT> 
  class DiscretisedDensity;
class ProjData;
class Succeeded;

//! transform image data
/*! \ingroup motion
    Currently only supports VoxelOnCartesianGrid image.
    Uses (tri)linear interpolation.
*/
Succeeded
transform_3d_object(DiscretisedDensity<3,float>& out_density, 
		    const DiscretisedDensity<3,float>& in_density, 
		    const RigidObject3DTransformation& rigid_object_transformation);

//! transform projection data
/*! \ingroup motion
    Currently only supports non-arccorrected data.
    
    Uses all available input segments
*/
Succeeded
transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation);

//! transform projection data
/*! \ingroup motion
    Currently only supports non-arccorrected data.
*/
Succeeded
transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation,
		    const int min_in_segment_num_to_process,
		    const int max_in_segment_num_to_process);

END_NAMESPACE_STIR
