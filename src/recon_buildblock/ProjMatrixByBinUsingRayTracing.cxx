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

/* History

   KT
   added registry things
   KT 21/02/2002
   added option for square FOV
   KT 15/05/2002 
   added possibility of multiple LORs in tangential direction
   call ProjMatrixByBin's new parsing functions
   KT 28/06/02 
   added option to take actual detector boundaries into account
 */

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/round.h"
#include <algorithm>
#include <math.h>

#ifndef STIR_NO_NAMESPACE
using std::min;
using std::max;
#endif
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
  ProjMatrixByBin::initialise_keymap();
  parser.add_start_key("Ray Tracing Matrix Parameters");
  parser.add_key("restrict to cylindrical FOV", &restrict_to_cylindrical_FOV);
  parser.add_key("number of rays in tangential direction to trace for each bin",
                  &num_tangential_LORs);
  parser.add_key("use actual detector boundaries", &use_actual_detector_boundaries);
  parser.add_key("do_symmetry_90degrees_min_phi", &do_symmetry_90degrees_min_phi);
  parser.add_key("do_symmetry_180degrees_min_phi", &do_symmetry_180degrees_min_phi);
  parser.add_stop_key("End Ray Tracing Matrix Parameters");
}


void
ProjMatrixByBinUsingRayTracing::set_defaults()
{
  ProjMatrixByBin::set_defaults();
  restrict_to_cylindrical_FOV = true;
  num_tangential_LORs = 1;
  use_actual_detector_boundaries = false;
  do_symmetry_90degrees_min_phi = true;
  do_symmetry_180degrees_min_phi = true;
}


bool
ProjMatrixByBinUsingRayTracing::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;
  if (num_tangential_LORs<1)
  { 
    warning("ProjMatrixByBinUsingRayTracing: num_tangential_LORs should be at least 1, but is %d\n",
            num_tangential_LORs);
    return true;
  }
  return false;
}

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
                                                density_info_ptr,
                                                do_symmetry_90degrees_min_phi,
                                                do_symmetry_180degrees_min_phi);
  const float sampling_distance_of_adjacent_LORs_xy =
    proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));
  
  if(sampling_distance_of_adjacent_LORs_xy/num_tangential_LORs > voxel_size.x() + 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy/num_tangential_LORs > voxel_size.y() + 1.E-3)
     warning("WARNING: ProjMatrixByBinUsingRayTracing used for pixel size (in x,y) "
             "that is smaller than the bin size divided by num_tangential_LORs.\n"
             "This matrix will completely miss some voxels for some (or all) views.\n");
  if(sampling_distance_of_adjacent_LORs_xy < voxel_size.x() - 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy < voxel_size.y() - 1.E-3)
     warning("WARNING: ProjMatrixByBinUsingRayTracing used for pixel size (in x,y) "
             "that is larger than the bin size.\n"
             "Backprojecting with this matrix might have artefacts at views 0 and 90 degrees.\n");

  if (use_actual_detector_boundaries)
    {
      const ProjDataInfoCylindricalNoArcCorr * proj_data_info_cyl_ptr =
	dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr.get());
      if (proj_data_info_cyl_ptr== 0)
	{
	  warning("ProjMatrixByBinUsingRayTracing: use_actual_detector_boundaries"
		  " is reset to false as the projection data should be non-arccorected.\n");
	  use_actual_detector_boundaries = false;
	}
      else 
	{
	  bool nocompression = 
	    proj_data_info_cyl_ptr->get_view_mashing_factor()==1;
	  for (int segment_num=proj_data_info_cyl_ptr->get_min_segment_num();
	       nocompression && segment_num <= proj_data_info_cyl_ptr->get_max_segment_num();
	       ++segment_num)
	    nocompression= 
	      proj_data_info_cyl_ptr->get_min_ring_difference(segment_num) ==
	      proj_data_info_cyl_ptr->get_max_ring_difference(segment_num);
	
	  if (!nocompression)
	    {
	      warning("ProjMatrixByBinUsingRayTracing: use_actual_detector_boundaries"
		      " is reset to false as the projection data as either mashed or uses axial compression\n");
	      use_actual_detector_boundaries = false;
	    }
	}

      if (use_actual_detector_boundaries)
	warning("ProjMatrixByBinUsingRayTracing: use_actual_detector_boundaries==true\n");

    }  
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


template <typename T>
static inline int sign(const T& t) 
{
  return t<0 ? -1 : 1;
}

// just do 1 LOR, returns true if lor is not empty
static void
ray_trace_one_lor(ProjMatrixElemsForOneBin& lor, 
                  const float s_in_mm, const float t_in_mm, 
                  const float cphi, const float sphi, 
                  const float costheta, const float tantheta, 
                  const float offset_in_z,
                  const float fovrad_in_mm,
                  const CartesianCoordinate3D<float>& voxel_size,
                  const bool restrict_to_cylindrical_FOV,
                  const int num_LORs)
{
  assert(lor.size() == 0);

  /* Find Intersection points of LOR and image FOV (assuming infinitely long scanner)*/
  /* (in voxel units) */
  CartesianCoordinate3D<float> start_point;  
  CartesianCoordinate3D<float> stop_point;
  {
    /* parametrisation of LOR is
         X= s*cphi + a*sphi, 
         Y= s*sphi - a*cphi, 
         Z= t/costheta+offset_in_z - a*tantheta
       find now min_a, max_a such that end-points intersect border of FOV 
    */
    float max_a;
    float min_a;
    
    if (restrict_to_cylindrical_FOV)
    {
      if (fabs(s_in_mm) >= fovrad_in_mm) return;
      // a has to be such that X^2+Y^2 == fovrad^2      
      max_a = sqrt(square(fovrad_in_mm) - square(s_in_mm));
      min_a = -max_a;
    } // restrict_to_cylindrical_FOV
    else
    {
      // use FOV which is square.
      // note that we use square and not rectangular as otherwise symmetries
      // would take us out of the FOV. TODO
      /*
        a has to be such that 
        |X| <= fovrad_in_mm &&  |Y| <= fovrad_in_mm
      */
      if (fabs(cphi) < 1.E-3 || fabs(sphi) < 1.E-3) 
      {
        if (fovrad_in_mm < fabs(s_in_mm))
          return;
        max_a = fovrad_in_mm;
        min_a = -fovrad_in_mm;
      }
      else
      {
        max_a = min((fovrad_in_mm*sign(sphi) - s_in_mm*cphi)/sphi,
                    (fovrad_in_mm*sign(cphi) + s_in_mm*sphi)/cphi);
        min_a = max((-fovrad_in_mm*sign(sphi) - s_in_mm*cphi)/sphi,
                    (-fovrad_in_mm*sign(cphi) + s_in_mm*sphi)/cphi);
        if (min_a > max_a - 1.E-3*voxel_size.x())
          return;
      }
      
    } //!restrict_to_cylindrical_FOV
    
    start_point.x() = (s_in_mm*cphi + max_a*sphi)/voxel_size.x();
    start_point.y() = (s_in_mm*sphi - max_a*cphi)/voxel_size.y(); 
    start_point.z() = (t_in_mm/costheta+offset_in_z - max_a*tantheta)/voxel_size.z();
    stop_point.x() = (s_in_mm*cphi + min_a*sphi)/voxel_size.x();
    stop_point.y() = (s_in_mm*sphi - min_a*cphi)/voxel_size.y(); 
    stop_point.z() = (t_in_mm/costheta+offset_in_z - min_a*tantheta)/voxel_size.z();
  
    // do actual ray tracing for this LOR
    
    RayTraceVoxelsOnCartesianGrid(lor, start_point, stop_point, voxel_size,
#ifdef NEWSCALE
           1.F/num_LORs // normalise to mm
#else
           1/voxel_size.x()/num_LORs // normalise to some kind of 'pixel units'
#endif
           );

    return;
  }

}
//////////////////////////////////////
void 
ProjMatrixByBinUsingRayTracing::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  const Bin bin = lor.get_bin();
  assert(bin.segment_num() >= proj_data_info_ptr->get_min_segment_num());    
  assert(bin.segment_num() <= proj_data_info_ptr->get_max_segment_num());    

  assert(lor.size() == 0);
   
  float phi;
  float s_in_mm = proj_data_info_ptr->get_s(bin);
  /* Implementation note.
     KT initialised s_in_mm above instead of in the if because this meant
     that gcc 3.0.1 generated identical results to the previous version of this file.
     Otherwise, some pixels at the boundary appear to be treated differently
     (probably due to different floating point rounding errors), at least
     on Linux on x86.
     A bit of a mistery that.
  */
  if (!use_actual_detector_boundaries)
  {
    phi = proj_data_info_ptr->get_phi(bin);
    //s_in_mm = proj_data_info_ptr->get_s(bin);
  }
  else
  {
    // can be static_cast later on
    const ProjDataInfoCylindricalNoArcCorr& proj_data_info_noarccor =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);
    // TODO check on 180 degrees for views
    const int num_detectors =
      proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring();
    const float ring_radius =
      proj_data_info_ptr->get_scanner_ptr()->get_ring_radius();

    int det_num1=0, det_num2=0;
    proj_data_info_noarccor.
      get_det_num_pair_for_view_tangential_pos_num(det_num1,
						 det_num2,
						 bin.view_num(),
						 bin.tangential_pos_num());
    phi = (det_num1+det_num2)*_PI/num_detectors-_PI/2;
    const float old_phi=proj_data_info_ptr->get_phi(bin);
    if (fabs(phi-old_phi)>2*_PI/num_detectors)
      warning("view %d old_phi %g new_phi %g\n",bin.view_num(), old_phi, phi);

    s_in_mm = ring_radius*sin((det_num1-det_num2)*_PI/num_detectors+_PI/2);
    const float old_s_in_mm=proj_data_info_ptr->get_s(bin);
    if (fabs(s_in_mm-old_s_in_mm)>proj_data_info_ptr->get_sampling_in_s(bin)*.0001)
      warning("tangential_pos_num %d old_s_in_mm %g new_s_in_mm %g\n",bin.tangential_pos_num(), old_s_in_mm, s_in_mm);

  }
  
  const float cphi = cos(phi);
  const float sphi = sin(phi);
  
  const float tantheta = proj_data_info_ptr->get_tantheta(bin);
  const float costheta = 1/sqrt(1+square(tantheta));
  const float t_in_mm = proj_data_info_ptr->get_t(bin);
   
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
  // KT 15/05/2002 move +(max_index.z()+min_index.z())/2.F offset here instead of in formulas for Z1f,Z2f
  const float offset_in_z = 
    (num_lors_per_axial_pos == 1 || tantheta == 0 ? 
     0.F : -sampling_distance_of_adjacent_LORs_z/4)
    - origin.z()
    +(max_index.z()+min_index.z())/2.F * voxel_size.z();

  // use FOV which is slightly 'inside' the image to avoid
  // index out of range
  const float fovrad_in_mm = 
    min((min(max_index.x(), -min_index.x())-1)*voxel_size.x(),
        (min(max_index.y(), -min_index.y())-1)*voxel_size.y()); 

  if (num_tangential_LORs == 1)
  {
    ray_trace_one_lor(lor, s_in_mm, t_in_mm, 
                        cphi, sphi, costheta, tantheta, 
                        offset_in_z, fovrad_in_mm, 
                        voxel_size,
                        restrict_to_cylindrical_FOV,
                        num_lors_per_axial_pos);    
  }
  else
  {
    ProjMatrixElemsForOneBin ray_traced_lor;

    // get_sampling_in_s returns sampling in interleaved case
    // interleaved case has a sampling which is twice as high
    const float s_inc = 
       (!use_actual_detector_boundaries ? 1 : 2) *
        proj_data_info_ptr->get_sampling_in_s(bin)/num_tangential_LORs;
    float current_s_in_mm =
        s_in_mm - s_inc*(num_tangential_LORs-1)/2.F;
    for (int s_LOR_num=1; s_LOR_num<=num_tangential_LORs; ++s_LOR_num, current_s_in_mm+=s_inc)
    {
      ray_traced_lor.erase();
      ray_trace_one_lor(ray_traced_lor, current_s_in_mm, t_in_mm, 
                          cphi, sphi, costheta, tantheta, 
                          offset_in_z, fovrad_in_mm, 
                          voxel_size,
                          restrict_to_cylindrical_FOV,
                          num_lors_per_axial_pos*num_tangential_LORs);
      //std::cerr << "ray traced size " << ray_traced_lor.size() << std::endl;
      lor.merge(ray_traced_lor);
    }
  }
      
  // now add on other LORs in axial direction
  if ( num_lors_per_axial_pos>1 && lor.size()>0)
  {          
    assert(num_lors_per_axial_pos==2);
    if (tantheta==0 ) 
    { 
      assert(lor.begin()->coord1() == -origin.z()/voxel_size.z());
      add_adjacent_z(lor);
    }
    else
    { 
      merge_zplus1(lor);
    }

  } // if( num_lors_per_axial_pos>1)
  
}

// TODO these currently do NOT follow the requirement that
// after processing lor.sort() == lor

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

