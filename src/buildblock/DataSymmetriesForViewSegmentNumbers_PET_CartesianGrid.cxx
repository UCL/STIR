//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief non-inline implementations for class DataSymmetriesForBins_PET_CartesianGrid

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "DataSymmetriesForBins_PET_CartesianGrid.h"
#include "ProjDataInfoCylindrical.h"
#include "VoxelsOnCartesianGrid.h"


START_NAMESPACE_TOMO

  // find correspondence between axial_pos_num and image coordinates:
  // z = num_planes_per_axial_pos * axial_pos_num + axial_pos_to_z_offset
  // compute the offset by matching up the centre of the scanner 
  // in the 2 coordinate systems

static void 
find_relation_between_coordinate_systems(int& num_planes_per_scanner_ring,
                                         VectorWithOffset<int>& num_planes_per_axial_pos,
                                         VectorWithOffset<float>& axial_pos_to_z_offset,
                                         const ProjDataInfoCylindrical* proj_data_info_cyl_ptr,
                                         const DiscretisedDensityOnCartesianGrid<3,float> *  cartesian_grid_info_ptr)
                                              
{

  const int min_segment_num = proj_data_info_cyl_ptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_cyl_ptr->get_max_segment_num();

  num_planes_per_axial_pos = VectorWithOffset<int>(min_segment_num, max_segment_num);
  axial_pos_to_z_offset = VectorWithOffset<float>(min_segment_num, max_segment_num);

  // TODO and WARNING: get_grid_spacing()[1] is z()    
  const float image_plane_spacing = cartesian_grid_info_ptr->get_grid_spacing()[1];
    
  {    
    const float num_planes_per_scanner_ring_float = 
      proj_data_info_cyl_ptr->get_ring_spacing() / image_plane_spacing;
        
    num_planes_per_scanner_ring = static_cast<int>(num_planes_per_scanner_ring_float + 0.5);
    
    if (fabs(num_planes_per_scanner_ring_float - num_planes_per_scanner_ring) > 1.E-5)
    error("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing\
    equal to the ring spacing of the scanner divided by an integer. Sorry\n");
  }
  
  
  for (int segment_num=min_segment_num; segment_num<=max_segment_num; ++segment_num)  
  {
    { 
      const float 
        num_planes_per_axial_pos_float = 
        proj_data_info_cyl_ptr->get_axial_sampling(segment_num)/image_plane_spacing;
      
      num_planes_per_axial_pos[segment_num] = static_cast<int>(num_planes_per_axial_pos_float + 0.5);
      
      if (fabs(num_planes_per_axial_pos_float - num_planes_per_axial_pos[segment_num]) > 1.E-5)
        error("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing\
        equal to the axial sampling in the projection data divided by an integer. Sorry\n");
      
    }  
    
    const float delta = proj_data_info_cyl_ptr->get_average_ring_difference(segment_num);
    
    axial_pos_to_z_offset[segment_num] = 
      (cartesian_grid_info_ptr->get_max_z() + cartesian_grid_info_ptr->get_min_z())/2.F
      -
      (num_planes_per_axial_pos[segment_num]
       *(proj_data_info_cyl_ptr->get_max_axial_pos_num(segment_num)  
         + proj_data_info_cyl_ptr->get_min_axial_pos_num(segment_num))
       + num_planes_per_scanner_ring*delta)/2;
  }
}

DataSymmetriesForBins_PET_CartesianGrid::
DataSymmetriesForBins_PET_CartesianGrid
(
 const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
 const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr
)
  : DataSymmetriesForBins(proj_data_info_ptr)/*,
    image_info_ptr(image_info_ptr)*/
{
  if(dynamic_cast<ProjDataInfoCylindrical *>(proj_data_info_ptr.get()) == NULL)
    error("DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid constructed with wrong type of ProjDataInfo: %s\n",
      typeid(*proj_data_info_ptr).name());

  const DiscretisedDensityOnCartesianGrid<3,float> *
    cartesian_grid_info_ptr =
     dynamic_cast<const DiscretisedDensityOnCartesianGrid<3,float> *>
      (image_info_ptr.get());
  
  if (cartesian_grid_info_ptr == NULL)
    error("DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid constructed with wrong type of image info: %s\n",
      typeid(*image_info_ptr).name());


  num_views= proj_data_info_ptr->get_num_views();
  
  find_relation_between_coordinate_systems(num_planes_per_scanner_ring,
                                         num_planes_per_axial_pos,
                                         axial_pos_to_z_offset,
                                         static_cast<const ProjDataInfoCylindrical *>(proj_data_info_ptr.get()),
                                         cartesian_grid_info_ptr);
}

END_NAMESPACE_TOMO
