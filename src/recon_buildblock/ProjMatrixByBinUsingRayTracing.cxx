//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByBinUsingRayTracing

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
#include <algorithm>
#include <math.h>

START_NAMESPACE_STIR


const char * const 
ProjMatrixByBinUsingRayTracing::registered_name =
  "Ray Tracing";

ProjMatrixByBinUsingRayTracing::
ProjMatrixByBinUsingRayTracing()
{
  set_defaults();
}

void 
ProjMatrixByBinUsingRayTracing::initialise_keymap()
{
  parser.add_start_key("Ray Tracing Matrix Parameters");
  parser.add_stop_key("End Ray Tracing Matrix Parameters");
}


void
ProjMatrixByBinUsingRayTracing::set_defaults()
{}

#if 0
// static helper function
// check if a is an integer multiple of b
static bool is_multiple(const float a, const float b)
{
  return fabs(fmod(static_cast<double>(a), static_cast<double>(b))) > 1E-5;
}
#endif

void
ProjMatrixByBinUsingRayTracing::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinUsingRayTracing initialised with a wrong type of DiscretisedDensity\n");

 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);

  symmetries_ptr = 
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_ptr);
  const float sampling_distance_of_adjacent_LORs_xy =
    proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));
  
  if(sampling_distance_of_adjacent_LORs_xy > voxel_size.x() + 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy > voxel_size.y() + 1.E-3)
     warning("WARNING: ProjMatrixByBinUsingRayTracing used for pixel size (in x,y) "
             "that is smaller than the bin size.\n"
             "This matrix will completely miss some voxels for some (or all) views.\n");
  if(sampling_distance_of_adjacent_LORs_xy < voxel_size.x() - 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy < voxel_size.y() - 1.E-3)
     warning("WARNING: ProjMatrixByBinUsingRayTracing used for pixel size (in x,y) "
             "that is larger than the bin size.\n"
             "Backprojecting with this matrix will have artefacts at views 0 and 90 degrees.\n");
  
};

/* this is used when 
   (tantheta==0 && sampling_distance_of_adjacent_LORs_z==2*voxel_size.z())
  it adds two  adjacents z with their half value
  */
static void 
add_adjacent_z(ProjMatrixElemsForOneBin& lor);

/* Complicated business to add the same values at z+1
   while taking care that the (x,y,z) coordinates remain unique in the LOR.
  (If you copy the LOR somewhere else, you can simply use 
   ProjMatrixElemsForOneBin::merge())
*/         
static void merge_zplus1(ProjMatrixElemsForOneBin& lor);


//////////////////////////////////////
void 
ProjMatrixByBinUsingRayTracing::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  const Bin bin = lor.get_bin();
  //assert(bin.axial_pos_num() == 0);
  assert(bin.tangential_pos_num() >= 0);
  // KT 20/06/2001 removed assert as it might come from a related bin which has larger abs(tang_pos_num)
  //assert(bin.tangential_pos_num() <= proj_data_info_ptr->get_max_tangential_pos_num());  
  assert(bin.segment_num()  >= 0  );
  assert(bin.segment_num() <= proj_data_info_ptr->get_max_segment_num());    
  assert(bin.view_num() <= proj_data_info_ptr->get_num_views()/4);    
  assert(bin.view_num()>=0);

  assert(lor.size() == 0);
     
  const float tantheta = proj_data_info_ptr->get_tantheta(bin);
  const float costheta = 1/sqrt(1+square(tantheta));
  const float phi = proj_data_info_ptr->get_phi(bin);
  const float cphi = cos(phi);
  const float sphi = sin(phi);
  const float s_in_mm = proj_data_info_ptr->get_s(bin);
  const float t_in_mm = proj_data_info_ptr->get_t(bin);


  // use FOV which is circular, and is slightly 'inside' the image to avoid 
  // index out of range
  const float fovrad_in_mm   = 
    min((min(max_index.x(), -min_index.x())-1)*voxel_size.x(),
        (min(max_index.y(), -min_index.y())-1)*voxel_size.y()); 
  if (s_in_mm >= fovrad_in_mm) return;


  const float sampling_distance_of_adjacent_LORs_z =
    proj_data_info_ptr->get_sampling_in_t(bin)/costheta;
 

  // find number of LORs we have to take, such that we don't miss voxels
  // we have to subtract a tiny amount from the quotient, to avoid having too many LORs
  // solely due to numerical rounding errors
  const int num_lors_per_axial_pos = 
    static_cast<int>(ceil(sampling_distance_of_adjacent_LORs_z / voxel_size.z() - 1.E-3));

  assert(num_lors_per_axial_pos>0);
  // code below is currently restricted to 2 LORs
  assert(num_lors_per_axial_pos<=2);

  // merging code assumes integer multiple
  assert(fabs(sampling_distance_of_adjacent_LORs_z 
              - num_lors_per_axial_pos*voxel_size.z()) <= 1E-4);


  // find offset in z, taking into account if there are 1 or 2 LORs
  // KT 20/06/2001 take origin.z() into account
  const float offset_in_z = 
    (num_lors_per_axial_pos == 1 || tantheta == 0 ? 
     0.F : -sampling_distance_of_adjacent_LORs_z/4)
    - origin.z();


  /* Intersection points of LOR and image FOV (assuming infinitely long scanner)*/
  /* compute   X1f, Y1f,,Z1f et al in voxelcoordinates. */
  
  const float TMP2f = sqrt(square(fovrad_in_mm) - square(s_in_mm));
  const float X2f = (s_in_mm * cphi - sphi * TMP2f)/voxel_size.x();
  const float X1f = (s_in_mm * cphi + sphi * TMP2f)/voxel_size.x();
  const float Y2f = (s_in_mm * sphi + cphi * TMP2f)/voxel_size.y();
  const float Y1f = (s_in_mm * sphi - cphi * TMP2f)/voxel_size.y();

  const float Z1f = 
    (t_in_mm/costheta - TMP2f*tantheta + offset_in_z)/voxel_size.z() 
    + (max_index.z() + min_index.z())/2.F;
  const float Z2f = 
    (t_in_mm/costheta + TMP2f*tantheta + offset_in_z)/voxel_size.z()
    + (max_index.z() + min_index.z())/2.F;

  const CartesianCoordinate3D<float> start_point(Z1f,Y1f,X1f);  
  const CartesianCoordinate3D<float> stop_point(Z2f,Y2f,X2f);  

  // do actual ray tracing for this LOR

  RayTraceVoxelsOnCartesianGrid(lor, start_point, stop_point, voxel_size,
#ifdef NEWSCALE
           1.F/num_lors_per_axial_pos // normalise to mm
#else
           1/voxel_size.x()/num_lors_per_axial_pos // normalise to some kind of 'pixel units'
#endif
           );

  // now add on other LORs
  if ( num_lors_per_axial_pos>1)
  {      
    
    assert(num_lors_per_axial_pos==2);
    if (tantheta==0 ) 
    { 
      assert(Z1f == -origin.z()/voxel_size.z());
      add_adjacent_z(lor);
    }
    else
    { 
      // lor.merge( lor2 );   
      merge_zplus1(lor);
    }

  } // if( num_lors_per_axial_pos>1)
  
}

// TODO these currently do NOT follow the requirement that
// after processing lor.sort() == before processing lor.sort()

static void 
add_adjacent_z(ProjMatrixElemsForOneBin& lor)
{
  // KT&SM 15/05/2000 bug fix !
  // first reserve enough memory for the whole vector
  // otherwise the iterators can be invalidated by memory allocation
  lor.reserve(lor.size() * 3);
  
  ProjMatrixElemsForOneBin::const_iterator element_ptr = lor.begin();
  ProjMatrixElemsForOneBin::const_iterator element_end = lor.end();
  
  while (element_ptr != element_end)
  {      
    lor.push_back( 
      ProjMatrixElemsForOneBin::value_type(
        Coordinate3D<int>(element_ptr->coord1()+1,element_ptr->coord2(),element_ptr->coord3()),element_ptr->get_value()/2));		 
    lor.push_back( 
      ProjMatrixElemsForOneBin::value_type(
        Coordinate3D<int>(element_ptr->coord1()-1,element_ptr->coord2(),element_ptr->coord3()),element_ptr->get_value()/2));
	   	   
    ++element_ptr;
  }
}


static void merge_zplus1(ProjMatrixElemsForOneBin& lor)
{
  // first reserve enough memory to keep everything. 
  // Otherwise iterators might be invalidated.
  lor.reserve(lor.size()*2);
 
  float next_value;
  float current_value = lor.begin()->get_value();
  ProjMatrixElemsForOneBin::const_iterator lor_old_end = lor.end();
  for (ProjMatrixElemsForOneBin::iterator lor_iter = lor.begin();
       lor_iter != lor_old_end; 
       ++lor_iter, current_value = next_value)
  {
    // save value before we potentially modify it below
    next_value = (lor_iter+1 == lor_old_end) ? 0.F : (lor_iter+1)->get_value();
    // check if we are the end, or the coordinates of the next voxel are
    // (x,y,z+1)
    if ((lor_iter+1 == lor_old_end) ||
      (lor_iter->coord3() != (lor_iter+1)->coord3()) || 
      (lor_iter->coord2() != (lor_iter+1)->coord2()) ||
      (lor_iter->coord1() + 1 != (lor_iter+1)->coord1()))
    {
      // if not, we can just push_back a new voxel
      lor.push_back(
         ProjMatrixElemsForOneBin::value_type(
           Coordinate3D<int>(lor_iter->coord1()+1, lor_iter->coord2(), lor_iter->coord3()), 
           current_value));
    }
    else
    {
      // increment value of next voxel with the current value
      *(lor_iter+1) += current_value;
    }
    
  }
  
}

END_NAMESPACE_STIR

