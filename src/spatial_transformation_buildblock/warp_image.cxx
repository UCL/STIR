//
/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.

 SPDX-License-Identifier: Apache-2.0

 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup buildblock
 \brief Implementation of function stir::warp_image
 \author Charalampos Tsoumpas
*/

#include "stir/spatial_transformation/warp_image.h"
#include "stir/error.h"

START_NAMESPACE_STIR
// using namespace BSpline;

VoxelsOnCartesianGrid<float>
warp_image(const shared_ptr<DiscretisedDensity<3, float>>& density_sptr,
           const shared_ptr<DiscretisedDensity<3, float>>& motion_x_sptr,
           const shared_ptr<DiscretisedDensity<3, float>>& motion_y_sptr,
           const shared_ptr<DiscretisedDensity<3, float>>& motion_z_sptr,
           const BSpline::BSplineType spline_type,
           const bool extend_borders)
{
  const DiscretisedDensityOnCartesianGrid<3, float>* density_cartesian_sptr
      = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>*>(density_sptr.get());
  const BasicCoordinate<3, float> grid_spacing = density_cartesian_sptr->get_grid_spacing();
  const CartesianCoordinate3D<float> origin = density_cartesian_sptr->get_origin();
  const BSpline::BSplinesRegularGrid<3, float> density_interpolation(*density_sptr, spline_type);

  BasicCoordinate<3, int> min;
  BasicCoordinate<3, int> max;
  const IndexRange<3> range = density_sptr->get_index_range();
  if (!range.get_regular_range(min, max))
    error("image is not in regular grid.\n");
  const BasicCoordinate<3, int> out_min = min;
  const BasicCoordinate<3, int> out_max = max;
#if 0
  if (extend_borders==1) {
    out_min[1]-=std::abs(voxel_shift_z) ;
    out_min[2]-=std::abs(voxel_shift_y) ;
    out_min[3]-=std::abs(voxel_shift_x) ;
    out_max[1]+=std::abs(voxel_shift_z) ;
    out_max[2]+=std::abs(voxel_shift_y) ;
    out_max[3]+=std::abs(voxel_shift_x) ;
  }
#endif
  const IndexRange<3> out_range(out_min, out_max);
  VoxelsOnCartesianGrid<float> out_density(out_range, origin, grid_spacing);

  BasicCoordinate<3, int> c;
  BasicCoordinate<3, double> d, l;
  for (c[1] = min[1]; c[1] <= max[1]; ++c[1])
    for (c[2] = min[2]; c[2] <= max[2]; ++c[2])
      for (c[3] = min[3]; c[3] <= max[3]; ++c[3])
        {
          l[1] = static_cast<double>((*motion_z_sptr)[c] / grid_spacing[1]);
          l[2] = static_cast<double>((*motion_y_sptr)[c] / grid_spacing[2]);
          l[3] = static_cast<double>((*motion_x_sptr)[c] / grid_spacing[3]);
          d[1] = static_cast<double>(c[1]) + l[1]; // for the IRTK version I had c-l, but for Christian's it seems to work as c+l
          d[2] = static_cast<double>(c[2]) + l[2];
          d[3] = static_cast<double>(c[3]) + l[3];
          // Temporary fix such that when radioactivity comes from outside is set to 0.
          // To fix this properly we need to modify the B-Splines interpolation method by changing the periodicity extrapolation.
          if ((d[1] <= static_cast<double>(min[1])) || (d[1] >= static_cast<double>(max[1]))
              || // I'm not considering the last plane if linear
              (d[2] <= static_cast<double>(min[2])) || (d[2] >= static_cast<double>(max[2]))
              || // because it's going to use extrapolated data
              (d[3] <= static_cast<double>(min[3]))
              || (d[3] >= static_cast<double>(max[3]))) // I haven't implemented anything for higher order
            out_density[c] = 0.F;
          else
            out_density[c] = density_interpolation(d);
        }
  return out_density;
}

END_NAMESPACE_STIR
