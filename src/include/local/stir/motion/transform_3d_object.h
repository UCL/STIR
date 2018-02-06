//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_motion_transformed_3d_object_H__
#define __stir_motion_transformed_3d_object_H__
/*!

  \file
  \ingroup motion
  \brief Declaration of functions to re-interpolate an image or projection data 
  to a new coordinate system.

  \author Kris Thielemans

*/

#include "stir/common.h"

// for caching functions
#include <utility>
#include "stir/Array.h"
#include "stir/assign.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

class RigidObject3DTransformation;
template <int num_dimensions, class elemT>
  class ObjectTransformation;

template <int num_dimensions, typename elemT> 
  class DiscretisedDensity;
class ProjData;
class Succeeded;


//! transform image data
/*! \ingroup motion
  \brief the interpolator has to push values from the input into the output image

 \warning \a transformation_in_to_out has to transform from in to out
 \see PushTransposeLinearInterpolator for an example interpolator.

 \par Developer's note
 This function is inline to avoid template instantiation problems.
 */
template <class ObjectTransformationT, class PushInterpolatorT>
inline
Succeeded 
transform_3d_object_push_interpolation(DiscretisedDensity<3,float>& out_density, 
				       const DiscretisedDensity<3,float>& in_density, 
				       const ObjectTransformationT& transformation_in_to_out,
				       const PushInterpolatorT& interpolator,
				       const bool do_jacobian);

//! transform image data
/*! \ingroup motion
  \brief the interpolator has to pull values from the input into the output image

 \warning \a transformation_out_to_in has to transform from out to in
 \see PullLinearInterpolator for an example interpolator.

 \par Developer's note
 This function is inline to avoid template instantiation problems.
 */
template <class ObjectTransformationT, class PullInterpolatorT>
inline
Succeeded 
transform_3d_object_pull_interpolation(DiscretisedDensity<3,float>& out_density, 
				       const DiscretisedDensity<3,float>& in_density, 
				       const ObjectTransformationT& transformation_out_to_in,
				       const PullInterpolatorT& interpolator,
				       const bool do_jacobian);

//! transform image data
/*! \ingroup motion
    Currently only supports VoxelOnCartesianGrid image.
    Uses (tri)linear interpolation.

    \todo cannot use ObjectTransformation yet as it needs the inverse transformation
*/
Succeeded
transform_3d_object(DiscretisedDensity<3,float>& out_density, 
		    const DiscretisedDensity<3,float>& in_density, 
		    const RigidObject3DTransformation& transformation_in_to_out);
		    //		    const ObjectTransformation<3,float>& transformation_in_to_out);


// ugly functions for storing transformed points.
// TODO clean up at some point

Array<3, BasicCoordinate<3,float> >
find_grid_coords_of_transformed_centres(const DiscretisedDensity<3,float>& source_density, 
					const DiscretisedDensity<3,float>& target_density, 
					const ObjectTransformation<3,float>& transformation_source_to_target);

// TODO need this for now to get Array<std::pair<>> to work
template <class T2>
  inline 
  void assign(std::pair<BasicCoordinate<3,float>, float>& x, T2 y)
{
  BasicCoordinate<3,float> tmp;
  assign(tmp,y);
  x = std::make_pair(tmp,float(y));
}

Array<3, std::pair<BasicCoordinate<3,float>, float> >
find_grid_coords_of_transformed_centres_and_jacobian(const DiscretisedDensity<3,float>& source_density, 
						     const DiscretisedDensity<3,float>& target_density, 
						     const ObjectTransformation<3,float>& transformation_source_to_target);

//! transform projection data
/*! \ingroup motion
    Currently only supports non-arccorrected data.
    
    Uses all available input segments
*/
Succeeded
transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& object_transformation);

//! transform image data using transposed matrix
/*! \ingroup motion
    Implements the transpose (not the inverse) of 
    transform_3d_object(DiscretisedDensity<3,float>&, const DiscretisedDensity<3,float>&,const RigidObject3DTransformation&)
    \todo cannot use ObjectTransformation yet as it needs the inverse transformation
*/
Succeeded
transpose_of_transform_3d_object(DiscretisedDensity<3,float>& out_density, 
				 const DiscretisedDensity<3,float>& in_density, 
				 const RigidObject3DTransformation& transformation_in_to_out);
//		    const ObjectTransformation<3,float>& transformation_in_to_out);

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

#include "local/stir/motion/transform_3d_object.inl"

#endif
