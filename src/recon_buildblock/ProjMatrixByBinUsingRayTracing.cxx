/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2011, Hammersmith Imanet Ltd
    Copyright (C) 2013-2014, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjMatrixByBinUsingRayTracing

  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
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
   KT 25/09/03
   allow disabling more symmetries
   allow smaller z- voxel sizes (but z-sampling in the projdata still has to 
   be an integer multiple of the z-voxel size).

 */

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/round.h"
#include "stir/modulo.h"
#include "stir/stream.h"
#include <algorithm>
#include <math.h>
#include <boost/format.hpp>

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
//******************** parsing *************

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
  parser.add_key("do_symmetry_swap_segment", &do_symmetry_swap_segment);
  parser.add_key("do_symmetry_swap_s", &do_symmetry_swap_s);
  parser.add_key("do_symmetry_shift_z", &do_symmetry_shift_z);
  parser.add_stop_key("End Ray Tracing Matrix Parameters");
}


void
ProjMatrixByBinUsingRayTracing::set_defaults()
{
  ProjMatrixByBin::set_defaults();
  this->restrict_to_cylindrical_FOV = true;
  this->num_tangential_LORs = 1;
  this->use_actual_detector_boundaries = false;
  this->do_symmetry_90degrees_min_phi = true;
  this->do_symmetry_180degrees_min_phi = true;
  this->do_symmetry_swap_segment = true;
  this->do_symmetry_swap_s = true;
  this->do_symmetry_shift_z = true;
  this->already_setup = false;
}


bool
ProjMatrixByBinUsingRayTracing::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;
  if (this->num_tangential_LORs<1)
  { 
    warning(boost::format("ProjMatrixByBinUsingRayTracing: num_tangential_LORs should be at least 1, but is %d")
            % this->num_tangential_LORs);
    return true;
  }
  this->already_setup = false;
  return false;
}

//******************** get/set pairs *************

bool
ProjMatrixByBinUsingRayTracing::
get_restrict_to_cylindrical_FOV() const
{
  return this->restrict_to_cylindrical_FOV;
}

void
ProjMatrixByBinUsingRayTracing::
set_restrict_to_cylindrical_FOV(bool val)
{
  this->already_setup = (this->restrict_to_cylindrical_FOV == val);
  this->restrict_to_cylindrical_FOV = val;
}

int
ProjMatrixByBinUsingRayTracing::
get_num_tangential_LORs() const
{
  return this->num_tangential_LORs;
}

void
ProjMatrixByBinUsingRayTracing::
set_num_tangential_LORs(int val)
{
  this->already_setup = (this->num_tangential_LORs == val);
  this->num_tangential_LORs = val;
}

bool
ProjMatrixByBinUsingRayTracing::
get_use_actual_detector_boundaries() const
{
  return this->use_actual_detector_boundaries;
}

void
ProjMatrixByBinUsingRayTracing::
set_use_actual_detector_boundaries(bool val)
{
  this->already_setup = (this->use_actual_detector_boundaries == val);
  this->use_actual_detector_boundaries = val;
}

bool
ProjMatrixByBinUsingRayTracing::
get_do_symmetry_90degrees_min_phi() const
{
  return this->do_symmetry_90degrees_min_phi;
}

void
ProjMatrixByBinUsingRayTracing::
set_do_symmetry_90degrees_min_phi(bool val)
{
  this->already_setup = (this->do_symmetry_90degrees_min_phi == val);
  this->do_symmetry_90degrees_min_phi = val;
}


bool
ProjMatrixByBinUsingRayTracing::
get_do_symmetry_180degrees_min_phi() const
{
  return this->do_symmetry_180degrees_min_phi;
}

void
ProjMatrixByBinUsingRayTracing::
set_do_symmetry_180degrees_min_phi(bool val)
{
  this->already_setup = (this->do_symmetry_180degrees_min_phi == val);
  this->do_symmetry_180degrees_min_phi = val;
}


bool
ProjMatrixByBinUsingRayTracing::
get_do_symmetry_swap_segment() const
{
  return this->do_symmetry_swap_segment;
}

void
ProjMatrixByBinUsingRayTracing::
set_do_symmetry_swap_segment(bool val)
{
  this->already_setup = (this->do_symmetry_swap_segment == val);
  this->do_symmetry_swap_segment = val;
}


bool
ProjMatrixByBinUsingRayTracing::
get_do_symmetry_swap_s() const
{
  return this->do_symmetry_swap_s;
}

void
ProjMatrixByBinUsingRayTracing::
set_do_symmetry_swap_s(bool val)
{
  this->already_setup = (this->do_symmetry_swap_s == val);
  this->do_symmetry_swap_s = val;
}


bool
ProjMatrixByBinUsingRayTracing::
get_do_symmetry_shift_z() const
{
  return this->do_symmetry_shift_z;
}

void
ProjMatrixByBinUsingRayTracing::
set_do_symmetry_shift_z(bool val)
{
  this->already_setup = (this->do_symmetry_shift_z == val);
  this->do_symmetry_shift_z = val;
}


//******************** actual implementation *************

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
  ProjMatrixByBin::set_up(proj_data_info_ptr_v, density_info_ptr);

  proj_data_info_ptr= proj_data_info_ptr_v; 
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinUsingRayTracing initialised with a wrong type of DiscretisedDensity\n");
 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);

  symmetries_sptr.reset(
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_ptr,
                                                do_symmetry_90degrees_min_phi,
                                                do_symmetry_180degrees_min_phi,
                                                do_symmetry_swap_segment,
                                                do_symmetry_swap_s,
                                                do_symmetry_shift_z));
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

#if 0
  // test if our 2D code does not have problems
  {
    // currently 2D code relies on the LOR falling in the middle of a voxel (in z-direction)
    const float z_shift = - origin.z()/voxel_size.z()
      +(max_index.z()+min_index.z())/2.F;
    if (fabs(z_shift - round(z_shift)) > .01)
      error("ProjMatrixByBinUsingRayTracing can currently not handle this image.\n"
            "Make sure you either have \n"
            "- an odd number of planes and z_origin=n* z_voxel_size\n"
            "- or an even number of planes and z_origin=(n+1/2)*z_voxel_size\n"
            "(for some integer n).\n");
  }
#endif


  this->already_setup = true;
  this->clear_cache();
};

/* this is used when 
   (tantheta==0 && sampling_distance_of_adjacent_LORs_z==2*voxel_size.z())
  it adds two  adjacents z with their half value
  */
static void 
add_adjacent_z(ProjMatrixElemsForOneBin& lor, 
               const float z_of_first_voxel, 
               const float right_edge_of_TOR);

#if 0
/* Complicated business to add the same values at z+1
   while taking care that the (x,y,z) coordinates remain unique in the LOR.
  (If you copy the LOR somewhere else, you can simply use 
   ProjMatrixElemsForOneBin::merge())
*/         
static void merge_zplus1(ProjMatrixElemsForOneBin& lor);
#endif

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
#ifdef STIR_PMRT_LARGER_FOV
      if (fabs(s_in_mm) >= fovrad_in_mm) return;
#else
      if (fabs(s_in_mm) > fovrad_in_mm) return;
#endif
      // a has to be such that X^2+Y^2 == fovrad^2      
      if (fabs(s_in_mm) == fovrad_in_mm) 
        {
          max_a = min_a = 0;
        }
      else
        {
          max_a = sqrt(square(fovrad_in_mm) - square(s_in_mm));
          min_a = -max_a;
        }
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

#if 0
    // KT 18/05/2005 this is no longer necessary

    // check we're not exactly at the border of 2 planes in the 2D case
    if (tantheta==0)
      {
        assert(stop_point.z()==start_point.z());
        if (fabs(modulo(stop_point.z(),1.F)-.5)<.001)
          error("ProjMatrixByBinUsingRayTracing: ray tracing at the border between two z-planes\n");
      }
    if (cphi==0)
      {
        assert(stop_point.y()==start_point.y());
        if (fabs(modulo(stop_point.y(),1.F)-.5)<.001)
          error("ProjMatrixByBinUsingRayTracing: ray tracing at the border between two y-planes\n");
      }
    if (sphi==0)
      {
        assert(stop_point.x()==start_point.x());
        if (fabs(modulo(stop_point.x(),1.F)-.5)<.001)
          error("ProjMatrixByBinUsingRayTracing: ray tracing at the border between two y-planes\n");
      }
#endif

    // find out in which direction we should do the ray tracing to obtain a sorted lor
    // we want to go from small z to large z, 
    // or if z are equal, from small y to large y and so on
    const bool from_start_to_stop =
      start_point.z() < stop_point.z() ||
      (start_point.z() == stop_point.z() &&
       (start_point.y() < stop_point.y() ||
        (start_point.y() == stop_point.y() &&
         (start_point.x() <= stop_point.x()))));

    // do actual ray tracing for this LOR
    
    RayTraceVoxelsOnCartesianGrid(lor, 
                                  from_start_to_stop? start_point : stop_point,
                                  !from_start_to_stop? start_point : stop_point,
                                  voxel_size,
#ifdef NEWSCALE
                                  1.F/num_LORs // normalise to mm
#else
                                  1/voxel_size.x()/num_LORs // normalise to some kind of 'pixel units'
#endif
           );

#ifndef NDEBUG
    {
      // TODO output is still not sorted... why?

      //ProjMatrixElemsForOneBin sorted_lor = lor;
      //sorted_lor.sort();
      //assert(lor == sorted_lor);
      lor.check_state();
    }
#endif
    return;
  }

}
//////////////////////////////////////
void 
ProjMatrixByBinUsingRayTracing::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  if (!this->already_setup)
    {
      error("ProjMatrixByBinUsingRayTracing used before calling setup");
    }

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

     TODO this is maybe solved now by having more decent handling of 
     start and end voxels.
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
      proj_data_info_ptr->get_scanner_ptr()->get_effective_ring_radius();

    int det_num1=0, det_num2=0;
    proj_data_info_noarccor.
      get_det_num_pair_for_view_tangential_pos_num(det_num1,
                                                 det_num2,
                                                 bin.view_num(),
                                                 bin.tangential_pos_num());
    phi = static_cast<float>((det_num1+det_num2)*_PI/num_detectors-_PI/2);
    const float old_phi=proj_data_info_ptr->get_phi(bin);
    if (fabs(phi-old_phi)>2*_PI/num_detectors)
      warning("view %d old_phi %g new_phi %g\n",bin.view_num(), old_phi, phi);

    s_in_mm = static_cast<float>(ring_radius*sin((det_num1-det_num2)*_PI/num_detectors+_PI/2));
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
  // theta=0 assumes that centre of 1 voxel coincides with centre of bin.
  // TODO test
  //if (num_lors_per_axial_pos>1 && tantheta==0 num_lors_per_axial_pos%2==0);

  // merging code assumes integer multiple
  assert(fabs(sampling_distance_of_adjacent_LORs_z/voxel_size.z()
              - num_lors_per_axial_pos) <= 1E-4);


  // find offset in z, taking into account if there are 1 or more LORs
  // KT 20/06/2001 take origin.z() into account
  // KT 15/05/2002 move +(max_index.z()+min_index.z())/2.F offset here instead of in formulas for Z1f,Z2f
  /* Here is how we find the offset of the first ray:
     for only 1 ray, it is simply found by refering to the middle of the image
     minus the origin.z().
     For multiple rays, the following reasoning is followed.

     First we look at oblique rays.
     All rays should be the same distance from eachother, which is
     dz = sampling_distance_of_adjacent_LORs_z/num_lors_per_axial_pos.
     Then you have to make sure that the middle of the set of rays for this
     bin corresponds to the middle of the TOR, i.e. the offset given above for 1 ray.
     So, we put the rays from 
     -dz*(num_lors_per_axial_pos-1)/2
     to
     +dz*(num_lors_per_axial_pos-1)/2

     Now we look at direct rays (tantheta=0).We have to choose rays which
     do not go exactly on the edge of 2 planes as this would give unreliable 
     results due to rounding errors.
     In addition, we can give a weight to the rays according to how much the 
     voxel overlaps with the TOR (in axial direction).
     Note that RayTracing* now sorts this out itself, so we could dispense with this 
     complication here. However, we can do it slightly more efficient here as
     we might be using 2 rays for one ring.
  */
  const float z_position_of_first_LOR_wrt_centre_of_TOR =
    (-sampling_distance_of_adjacent_LORs_z/(2*num_lors_per_axial_pos)*
      (num_lors_per_axial_pos-1))
    - origin.z();
  float offset_in_z = 
    z_position_of_first_LOR_wrt_centre_of_TOR
    +(max_index.z()+min_index.z())/2.F * voxel_size.z();

  if (tantheta==0)
    {
      // make sure we don't ray-trace exactly between 2 planes
      // z-coordinate (in voxel units) will be
      //  (t_in_mm+offset_in_z)/voxel_size.z();
      // if so, we ray trace first to the voxels at smaller z, but will add the 
      // other plane later (in add_adjacent_z)
      if (fabs(modulo((t_in_mm+offset_in_z)/voxel_size.z(),1.F)-.5)<.001)
        offset_in_z -= .1F*voxel_size.z();
    }


  // use FOV which is slightly 'inside' the image to avoid
  // index out of range
#ifdef STIR_PMRT_LARGER_FOV
  const float fovrad_in_mm = 
    min((min(max_index.x(), -min_index.x())+.45F)*voxel_size.x(),
        (min(max_index.y(), -min_index.y())+.45F)*voxel_size.y()); 
#else
  const float fovrad_in_mm = 
    min((min(max_index.x(), -min_index.x()))*voxel_size.x(),
        (min(max_index.y(), -min_index.y()))*voxel_size.y()); 
#endif

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
  if (lor.size()>0)
  {          
    if (tantheta==0 ) 
      { 
        const float z_of_first_voxel=
          lor.begin()->coord1() +
          origin.z()/voxel_size.z() -
          (max_index.z() + min_index.z())/2.F;
        const float left_edge_of_TOR =
          (t_in_mm - sampling_distance_of_adjacent_LORs_z/2
           )/voxel_size.z();
        const float right_edge_of_TOR =
          (t_in_mm + sampling_distance_of_adjacent_LORs_z/2
           )/voxel_size.z();

        add_adjacent_z(lor, z_of_first_voxel - left_edge_of_TOR, right_edge_of_TOR -left_edge_of_TOR);
      }
    else if (num_lors_per_axial_pos>1)
      {
#if 0
        if (num_lors_per_axial_pos==2)
          {         
            merge_zplus1(lor);
          }
        else
#endif
          { 
            // make copy of LOR that will be used to add adjacent z
            ProjMatrixElemsForOneBin lor_with_next_z = lor;
            // reserve enough memory to avoid reallocations
            lor.reserve(lor.size()*num_lors_per_axial_pos);
            // now add adjacent z
            for (int z_index=1; z_index<num_lors_per_axial_pos; ++z_index)
              {
                // add 1 to each z in the LOR
                ProjMatrixElemsForOneBin::iterator element_ptr = lor_with_next_z.begin();
                const ProjMatrixElemsForOneBin::iterator element_end = lor_with_next_z.end();
                while (element_ptr != element_end)
                  {
                    *element_ptr = 
                      ProjMatrixElemsForOneBin::
                      value_type(
                                 Coordinate3D<int>(element_ptr->coord1()+1,
                                                   element_ptr->coord2(),
                                                   element_ptr->coord3()),
                                 element_ptr->get_value());
                    ++element_ptr;
                  }
                // now merge it into the original
                lor.merge(lor_with_next_z);
              }
          }
      } // if( tantheta!=0 && num_lors_per_axial_pos>1)
  } //if (lor.size()!=0)
  
}

static void 
add_adjacent_z(ProjMatrixElemsForOneBin& lor, 
               const float z_of_first_voxel, 
               const float right_edge_of_TOR)
{
  assert(lor.size()>0);
  assert(z_of_first_voxel+.5>=0);
  assert(z_of_first_voxel-.5<=right_edge_of_TOR);
  // first reserve enough memory for the whole vector
  // this speeds things up.
  const int num_overlapping_voxels =
    round(ceil(right_edge_of_TOR-z_of_first_voxel+.5));
  lor.reserve(lor.size() * num_overlapping_voxels);
  
  // point to end of original LOR, i.e. first plane
  // const ProjMatrixElemsForOneBin::const_iterator element_end = lor.end();
  const std::size_t org_size = lor.size();

  for (int z_index= 1; /* no end condition here */; ++z_index)
    {
      const float overlap_of_voxel_with_TOR =
        std::min(right_edge_of_TOR, z_of_first_voxel + z_index + .5F) -
        std::max(0.F, z_of_first_voxel + z_index - .5F);
      if (overlap_of_voxel_with_TOR<=0.0001) // check if beyond TOR or overlap too small to bother
        {
          assert(num_overlapping_voxels>=z_index);
          break;
        }
      assert(overlap_of_voxel_with_TOR < 1.0001);
      const int new_z = lor.begin()->coord1()+z_index;
      if (overlap_of_voxel_with_TOR>.9999) // test if it is 1
        {
          // just copy the value
          std::size_t count = 0; // counter for elements in original LOR
          for (  ProjMatrixElemsForOneBin::const_iterator element_ptr = lor.begin();
                 count != org_size; //element_ptr != element_end;
                 ++element_ptr, ++count)
            {      
              assert(lor.size()+1 <= lor.capacity()); // not really necessary now, but check on reserve()  best for performance
              assert(new_z == element_ptr->coord1()+z_index);
              lor.push_back(
                            ProjMatrixElemsForOneBin::
                            value_type(
                                       Coordinate3D<int>(new_z,
                                                         element_ptr->coord2(),
                                                         element_ptr->coord3()),
                                       element_ptr->get_value()));
            }
        }
      else
        {
          // multiply the value with the overlap
          std::size_t count = 0; // counter for elements in original LOR
          for (  ProjMatrixElemsForOneBin::const_iterator element_ptr = lor.begin();
                 count != org_size; //element_ptr != element_end;
                 ++element_ptr, ++count)
            {      
              assert(lor.size()+1 <= lor.capacity());
              assert(new_z == element_ptr->coord1()+z_index);
              lor.push_back(
                            ProjMatrixElemsForOneBin::
                            value_type(
                                       Coordinate3D<int>(new_z,
                                                         element_ptr->coord2(),
                                                         element_ptr->coord3()),
                                       element_ptr->get_value()*overlap_of_voxel_with_TOR));
            }
        }
    } // loop over z_index

  // now check original z
  {
    const float overlap_of_voxel_with_TOR =
      std::min(right_edge_of_TOR, z_of_first_voxel + .5F) -
      std::max(0.F, z_of_first_voxel - .5F);
    assert (overlap_of_voxel_with_TOR>0);
    assert(overlap_of_voxel_with_TOR < 1.0001);
    if (overlap_of_voxel_with_TOR<.9999) // test if it is 1
      {
        // multiply the value with the overlap
        std::size_t count = 0; // counter for elements in original LOR
        for (  ProjMatrixElemsForOneBin::iterator element_ptr = lor.begin();
               count != org_size; //element_ptr != element_end;
               ++element_ptr, ++count)
            *element_ptr *= overlap_of_voxel_with_TOR;
      }
  }
#ifndef NDEBUG
  {
    // ProjMatrixElemsForOneBin sorted_lor = lor;
    // sorted_lor.sort();
    // assert(lor == sorted_lor);
    lor.check_state();
  }
#endif
}

#if 0
/*
  This function add another image row (with z+1) to the LOR, with the
  same x,y and value.
  However, it only works properly if the original LOR is such that
  any voxels with identical x,y but z'-z=1 are adjacent in the LOR.
  A sufficient condition for this is
  - the z-coord is in ascending order
  - only one of x,y,z changes to go to the next voxel
  (the latter happens presumably for a single ray tracing, but not necessarily
   in general. Take for instance a TOR wider than the voxel.)

  If the above condition is not satisfied, the current implementation can end
  up with 1 voxel occuring more than once in the end result.
  
  This could easily be solved by checking this at the end (after a sort()).
  However, as we don't do this yet, we currently no longer call this function.
*/
static void merge_zplus1(ProjMatrixElemsForOneBin& lor)
{
  // first reserve enough memory to keep everything. 
  // Otherwise iterators might be invalidated.

  lor.reserve(lor.size()*2);

  cerr << "before merge\n";
#if 0
      ProjMatrixElemsForOneBin::const_iterator iter = lor.begin();
      while (iter!= lor.end())
        {
          std::cerr << iter->get_coords() 
                    << ':' << iter->get_value()
                    << '\n';
          ++iter;
        }
#endif


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
    // not (x,y,z+1)
    if ((lor_iter+1 == lor_old_end) ||
      (lor_iter->coord3() != (lor_iter+1)->coord3()) || 
      (lor_iter->coord2() != (lor_iter+1)->coord2()) ||
      (lor_iter->coord1() + 1 != (lor_iter+1)->coord1()))
    {
      // if so, we can just push_back a new voxel (but LOR won't be sorted though)
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
  cerr << "after merge\n";  
  lor.check_state();
  cerr << "after check_St\n";
}
#endif

END_NAMESPACE_STIR

