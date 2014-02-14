//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup motion
  \brief Functions to re-interpolate an image 

  \author Kris Thielemans

*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
START_NAMESPACE_STIR

template <class ObjectTransformationT, class PushInterpolatorT>
Succeeded 
transform_3d_object_push_interpolation(DiscretisedDensity<3,float>& out_density, 
				       const DiscretisedDensity<3,float>& in_density, 
				       const ObjectTransformationT& transformation_in_to_out,
				       const PushInterpolatorT& interpolator,
				       const bool do_jacobian)
{

  const VoxelsOnCartesianGrid<float>& in_image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const&>(in_density);
  VoxelsOnCartesianGrid<float>& out_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(out_density);

  interpolator.set_output(out_image);

  for (int z= in_image.get_min_index(); z<= in_image.get_max_index(); ++z)
    for (int y= in_image[z].get_min_index(); y<= in_image[z].get_max_index(); ++y)
      for (int x= in_image[z][y].get_min_index(); x<= in_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(z,y,x) * in_image.get_voxel_size() +
          in_image.get_origin();
        const CartesianCoordinate3D<float> new_point =
          transformation_in_to_out.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - out_image.get_origin()) / out_image.get_voxel_size();
	const float jacobian =
	  do_jacobian 
	  ?  transformation_in_to_out.jacobian(current_point)
	  : 1;
        interpolator.add_to(new_point_in_image_coords, in_image[z][y][x]*jacobian);
      }
  return Succeeded::yes;
}

template <class ObjectTransformationT, class PullInterpolatorT>
Succeeded 
transform_3d_object_pull_interpolation(DiscretisedDensity<3,float>& out_density, 
				       const DiscretisedDensity<3,float>& in_density, 
				       const ObjectTransformationT& transformation_out_to_in,
				       const PullInterpolatorT& interpolator,
				       const bool do_jacobian)
{

  const VoxelsOnCartesianGrid<float>& in_image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const&>(in_density);
  VoxelsOnCartesianGrid<float>& out_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(out_density);

  interpolator.set_input(in_density);

  for (int z= out_image.get_min_index(); z<= out_image.get_max_index(); ++z)
    for (int y= out_image[z].get_min_index(); y<= out_image[z].get_max_index(); ++y)
      for (int x= out_image[z][y].get_min_index(); x<= out_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(static_cast<float>(z),
				       static_cast<float>(y),
				       static_cast<float>(x)) * 
	  out_image.get_voxel_size() +
          out_image.get_origin();
        const CartesianCoordinate3D<float> new_point =
          transformation_out_to_in.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - in_image.get_origin()) / in_image.get_voxel_size();
	out_image[z][y][x] =
	  interpolator(new_point_in_image_coords);
	if (do_jacobian)
	  out_image[z][y][x] *= transformation_out_to_in.jacobian(current_point);

      }

  return Succeeded::yes;
}


END_NAMESPACE_STIR
