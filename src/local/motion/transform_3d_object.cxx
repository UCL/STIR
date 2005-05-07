//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup motion
  \brief Functions to re-interpolate an image or projection data to a new coordinate system.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "local/stir/motion/transform_3d_object.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/VectorWithOffset.h"
#include "stir/SegmentByView.h"
#include "stir/Bin.h"
#include "stir/shared_ptr.h"
#include "stir/round.h"
#include "stir/Succeeded.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#ifdef ROT_INT
#include "local/stir/motion/bin_interpolate.h"
#endif
START_NAMESPACE_STIR

Succeeded 
transform_3d_object(DiscretisedDensity<3,float>& out_density, 
    		    const DiscretisedDensity<3,float>& in_density, 
			  const RigidObject3DTransformation& rigid_object_transformation)
{

  const VoxelsOnCartesianGrid<float>& in_image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const&>(in_density);
  VoxelsOnCartesianGrid<float>& out_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>&>(out_density);

#if 0
  for (int z= in_image.get_min_index(); z<= in_image.get_max_index(); ++z)
    for (int y= in_image[z].get_min_index(); y<= in_image[z].get_max_index(); ++y)
      for (int x= in_image[z][y].get_min_index(); x<= in_image[z][y].get_max_index(); ++x)
      {
        const CartesianCoordinate3D<float> current_point =
          CartesianCoordinate3D<float>(z,y,x) * in_image.get_voxel_size() +
          in_image.get_origin();
        const CartesianCoordinate3D<float> new_point =
          rigid_object_transformation.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - out_image.get_origin()) / out_image.get_voxel_size();
        // now find nearest neighbour
        const CartesianCoordinate3D<int> 
           left_neighbour(round(floor(new_point_in_image_coords.z())),
                             floor(round(new_point_in_image_coords.y())),
                             floor(round(new_point_in_image_coords.x())));

        if (left_neighbour[1] <= out_image.get_max_index() &&
            left_neighbour[1] >= out_image.get_min_index() &&
            left_neighbour[2] <= out_image[left_neighbour[1]].get_max_index() &&
            left_neighbour[2] >= out_image[left_neighbour[1]].get_min_index() &&
            left_neighbour[3] <= out_image[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
            left_neighbour[3] >= out_image[left_neighbour[1]][left_neighbour[2]].get_min_index())
          out_image[left_neighbour[1]][left_neighbour[2]][left_neighbour[3]] +=
            in_image[z][y][x];
      }
#else

  const RigidObject3DTransformation inverse_rigid_object_transformation =
    rigid_object_transformation.inverse();

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
          inverse_rigid_object_transformation.transform_point(current_point);
        const CartesianCoordinate3D<float> new_point_in_image_coords =
           (new_point - in_image.get_origin()) / in_image.get_voxel_size();
        // now find left neighbour
        const CartesianCoordinate3D<int> 
           left_neighbour(round(floor(new_point_in_image_coords.z())),
                             round(floor(new_point_in_image_coords.y())),
                             round(floor(new_point_in_image_coords.x())));

        if (left_neighbour[1] < in_image.get_max_index() &&
            left_neighbour[1] > in_image.get_min_index() &&
            left_neighbour[2] < in_image[left_neighbour[1]].get_max_index() &&
            left_neighbour[2] > in_image[left_neighbour[1]].get_min_index() &&
            left_neighbour[3] < in_image[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
            left_neighbour[3] > in_image[left_neighbour[1]][left_neighbour[2]].get_min_index())
	  {
	    const int x1=left_neighbour[3];
	    const int y1=left_neighbour[2];
	    const int z1=left_neighbour[1];
	    const int x2=left_neighbour[3]+1;
	    const int y2=left_neighbour[2]+1;
	    const int z2=left_neighbour[1]+1;
	    const float ix = new_point_in_image_coords[3]-x1;
	    const float iy = new_point_in_image_coords[2]-y1;
	    const float iz = new_point_in_image_coords[1]-z1;
	    const float ixc = 1 - ix;
	    const float iyc = 1 - iy;
	    const float izc = 1 - iz;
	    out_image[z][y][x] =
	      ixc * (iyc * (izc * in_image[z1][y1][x1]
			    + iz  * in_image[z2][y1][x1])
		     + iy * (izc * in_image[z1][y2][x1]
			      + iz  * in_image[z2][y2][x1])) 
	      + ix * (iyc * (izc * in_image[z1][y1][x2]
			     + iz  * in_image[z2][y1][x2])
		      + iy * (izc * in_image[z1][y2][x2]
			      + iz  * in_image[z2][y2][x2]));
	  }
	else
	  out_image[z][y][x] = 0;

      }
#endif

  return Succeeded::yes;
}



Succeeded
transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation)
{
  return transform_3d_object(out_proj_data,
			     in_proj_data,
			     rigid_object_transformation,
			     in_proj_data.get_min_segment_num(),
			     in_proj_data.get_max_segment_num());			     
}

Succeeded
transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation,
		    const int min_in_segment_num_to_process,
		    const int max_in_segment_num_to_process)
{
#ifdef NEW_ROT
  warning( "Using NEW_ROT");
#else
  warning("Using original ROT");
#endif
#ifndef NEW_ROT
  const ProjDataInfoCylindricalNoArcCorr* const 
   out_proj_data_info_noarccor_ptr = 
       dynamic_cast<const ProjDataInfoCylindricalNoArcCorr* const>(out_proj_data.get_proj_data_info_ptr());
  const ProjDataInfoCylindricalNoArcCorr* const 
    in_proj_data_info_noarccor_ptr = 
       dynamic_cast<const ProjDataInfoCylindricalNoArcCorr* const>(in_proj_data.get_proj_data_info_ptr());
  if (out_proj_data_info_noarccor_ptr == 0 ||
      in_proj_data_info_noarccor_ptr == 0)
    {
      warning("Wrong type of proj_data_info (no-arccorrection)\n");
      return Succeeded::no;
    }
#else
  const ProjDataInfo&
   out_proj_data_info = 
       *out_proj_data.get_proj_data_info_ptr();
  const ProjDataInfo& 
    in_proj_data_info = *in_proj_data.get_proj_data_info_ptr();
#endif
  const int out_min_segment_num = out_proj_data.get_min_segment_num();
  const int out_max_segment_num = out_proj_data.get_max_segment_num();

#if 1

  warning("Using push interpolation");
#ifdef ROT_INT
  warning("with linear interpolation");
#endif
  VectorWithOffset<shared_ptr<SegmentByView<float> > > out_seg_ptr(out_min_segment_num, out_max_segment_num);
  for (int segment_num = out_min_segment_num;
       segment_num <= out_max_segment_num;
       ++segment_num)    
    out_seg_ptr[segment_num] = 
      new SegmentByView<float>(out_proj_data.get_empty_segment_by_view(segment_num));
  for (int segment_num = min_in_segment_num_to_process;
       segment_num <= max_in_segment_num_to_process;
       ++segment_num)    
    {       
      const SegmentByView<float> in_segment = 
        in_proj_data.get_segment_by_view( segment_num);
      std::cerr << "segment_num "<< segment_num << std::endl;
      const int in_max_ax_pos_num = in_segment.get_max_axial_pos_num();
      const int in_min_ax_pos_num = in_segment.get_min_axial_pos_num();
      const int in_max_view_num = in_segment.get_max_view_num();
      const int in_min_view_num = in_segment.get_min_view_num();
      const int in_max_tang_pos_num = in_segment.get_max_tangential_pos_num();
      const int in_min_tang_pos_num = in_segment.get_min_tangential_pos_num();
      for (int view_num=in_min_view_num; view_num<=in_max_view_num; ++view_num)
	for (int ax_pos_num=in_min_ax_pos_num; ax_pos_num<=in_max_ax_pos_num; ++ax_pos_num)
	  for (int tang_pos_num=in_min_tang_pos_num; tang_pos_num<=in_max_tang_pos_num; ++tang_pos_num)
	    {
	      Bin bin(segment_num, view_num, ax_pos_num, tang_pos_num,
		      in_segment[view_num][ax_pos_num][tang_pos_num]);
	      if (bin.get_bin_value()==0)
		continue;
#ifndef ROT_INT
	      rigid_object_transformation.transform_bin(bin,
# ifndef NEW_ROT
							*out_proj_data_info_noarccor_ptr,
							*in_proj_data_info_noarccor_ptr
# else
							out_proj_data_info,
							in_proj_data_info
# endif
							);
	      if (bin.get_bin_value()>0)
		(*out_seg_ptr[bin.segment_num()])[bin.view_num()]
						 [bin.axial_pos_num()]
						 [bin.tangential_pos_num()] +=
		  bin.get_bin_value();
#else
# ifndef NEW_ROT
# error ROT_INT defined but NEW_ROT not
# endif
	      LORInAxialAndNoArcCorrSinogramCoordinates<float> transformed_lor;
	      if (get_transformed_LOR(transformed_lor,
				      rigid_object_transformation,
				      bin,
				      in_proj_data_info) == Succeeded::yes)
		bin_interpolate(out_seg_ptr, transformed_lor, out_proj_data_info, in_proj_data_info, bin.get_bin_value());
#endif

	    }
    }
  Succeeded succes = Succeeded::yes;
  for (int segment_num = out_proj_data.get_min_segment_num();
       segment_num <= out_proj_data.get_max_segment_num();
       ++segment_num)    
    {       
      if (out_proj_data.set_segment(*out_seg_ptr[segment_num]) == Succeeded::no)
             succes = Succeeded::no;
    }

  return succes;


#else

  warning("Using pull interpolation");
  
  const RigidObject3DTransformation 
    inverse_rigid_object_transformation = 
    rigid_object_transformation.inverse();
  VectorWithOffset<shared_ptr<SegmentByView<float> > > 
    in_seg_ptr(min_in_segment_num_to_process,max_in_segment_num_to_process);
  for (int segment_num = min_in_segment_num_to_process;
       segment_num <= max_in_segment_num_to_process;
       ++segment_num)    
    in_seg_ptr[segment_num] = 
      new SegmentByView<float>(in_proj_data.get_segment_by_view(segment_num));
  for (int segment_num = out_min_segment_num;
       segment_num <= out_max_segment_num;
       ++segment_num)    
    {       
      SegmentByView<float> out_segment = 
        out_proj_data.get_empty_segment_by_view( segment_num);
      std::cerr << "segment_num "<< segment_num << std::endl;
      const int out_max_ax_pos_num = out_segment.get_max_axial_pos_num();
      const int out_min_ax_pos_num = out_segment.get_min_axial_pos_num();
      const int out_max_view_num = out_segment.get_max_view_num();
      const int out_min_view_num = out_segment.get_min_view_num();
      const int out_max_tang_pos_num = out_segment.get_max_tangential_pos_num();
      const int out_min_tang_pos_num = out_segment.get_min_tangential_pos_num();
      for (int view_num=out_min_view_num; view_num<=out_max_view_num; ++view_num)
	for (int ax_pos_num=out_min_ax_pos_num; ax_pos_num<=out_max_ax_pos_num; ++ax_pos_num)
	  for (int tang_pos_num=out_min_tang_pos_num; tang_pos_num<=out_max_tang_pos_num; ++tang_pos_num)
	    {
	      Bin bin(segment_num, view_num, ax_pos_num, tang_pos_num,1);
	      inverse_rigid_object_transformation.
		transform_bin(bin,
#ifndef NEW_ROT
			      *in_proj_data_info_noarccor_ptr,
			      *out_proj_data_info_noarccor_ptr
#else
			      in_proj_data_info,
			      out_proj_data_info
#endif
							);

	      if (bin.get_bin_value()>0 &&
		  bin.segment_num()>=min_in_segment_num_to_process &&
		  bin.segment_num()<=max_in_segment_num_to_process)
		{		  
		  out_segment[view_num][ax_pos_num][tang_pos_num] =
		    (*in_seg_ptr[bin.segment_num()])[bin.view_num()]
		    [bin.axial_pos_num()]
		    [bin.tangential_pos_num()];
		}
	    }
      if (out_proj_data.set_segment(out_segment) == Succeeded::no)
	return Succeeded::no;
    }

  return Succeeded::yes;
#endif


}

END_NAMESPACE_STIR
