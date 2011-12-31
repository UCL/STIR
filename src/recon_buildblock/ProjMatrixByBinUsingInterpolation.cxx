//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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

  \brief non-inline implementations for stir::ProjMatrixByBinUsingInterpolation

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/ProjMatrixByBinUsingInterpolation.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include <algorithm>
#include <math.h>

START_NAMESPACE_STIR

ProjMatrixByBinUsingInterpolation::JacobianForIntBP::
JacobianForIntBP(const ProjDataInfoCylindrical* proj_data_info_ptr, bool exact)
     
     : R2(square(proj_data_info_ptr->get_ring_radius())),
       ring_spacing2 (square(proj_data_info_ptr->get_ring_spacing())),
       arccor(dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(proj_data_info_ptr)!=0),
       backprojection_normalisation 
      (proj_data_info_ptr->get_ring_spacing()/2/proj_data_info_ptr->get_num_views()),
       use_exact_Jacobian_now(exact)
{
  assert(arccor ||
	 dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_ptr)!=0);
}

const char * const 
ProjMatrixByBinUsingInterpolation::registered_name =
  "Interpolation";

ProjMatrixByBinUsingInterpolation::
ProjMatrixByBinUsingInterpolation()
{
  set_defaults();
}

void 
ProjMatrixByBinUsingInterpolation::initialise_keymap()
{
  parser.add_start_key("Interpolation Matrix Parameters");
  parser.add_key("use_piecewise_linear_interpolation", &use_piecewise_linear_interpolation_now);
  parser.add_key("use_exact_Jacobian",&use_exact_Jacobian_now);
  ProjMatrixByBin::initialise_keymap();
  parser.add_key("do_symmetry_90degrees_min_phi", &do_symmetry_90degrees_min_phi);
  parser.add_key("do_symmetry_180degrees_min_phi", &do_symmetry_180degrees_min_phi);
  parser.add_key("do_symmetry_swap_segment", &do_symmetry_swap_segment);
  parser.add_key("do_symmetry_swap_s", &do_symmetry_swap_s);
  parser.add_key("do_symmetry_shift_z", &do_symmetry_shift_z);
  parser.add_stop_key("End Interpolation Matrix Parameters");
}


void
ProjMatrixByBinUsingInterpolation::set_defaults()
{
  ProjMatrixByBin::set_defaults();
  do_symmetry_90degrees_min_phi = true;
  do_symmetry_180degrees_min_phi = true;
  do_symmetry_swap_segment = true;
  do_symmetry_swap_s = true;
  do_symmetry_shift_z = true;

  use_piecewise_linear_interpolation_now = true;
  use_exact_Jacobian_now = true;
}


bool
ProjMatrixByBinUsingInterpolation::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;
  return false;
}


void
ProjMatrixByBinUsingInterpolation::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{
  proj_data_info_ptr= proj_data_info_ptr_v; 

  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinUsingInterpolation initialised with a wrong type of DiscretisedDensity\n");
 
  densel_range = image_info_ptr->get_index_range();
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  const float z_to_middle =
    (densel_range.get_max_index() + densel_range.get_min_index())*voxel_size.z()/2.F;
  origin.z() -= z_to_middle;

  symmetries_ptr.reset(
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_ptr,
                                                do_symmetry_90degrees_min_phi,
                                                do_symmetry_180degrees_min_phi,
						do_symmetry_swap_segment,
						do_symmetry_swap_s,
						do_symmetry_shift_z));

  if (dynamic_cast<const ProjDataInfoCylindrical*>(proj_data_info_ptr.get())==0)
    error("ProjMatrixByBinUsingInterpolation needs ProjDataInfoCylindrical for jacobian\n");
  jacobian = JacobianForIntBP(&(proj_data_info_cyl()), use_exact_Jacobian_now);

  // TODO assumes that all segments have span or not
  {
    const float relative_vox_sampling = 
      voxel_size.z() /
      proj_data_info_ptr->get_sampling_in_m(Bin(0,0,0,0));
    if (use_piecewise_linear_interpolation_now)
      {
	if (fabs(relative_vox_sampling-.5)<.01)
	  warning("Using piecewise-linear interpolation\n");
	else
	  {
	    warning("Switching OFF piecewise-linear interpolation\n");
	    use_piecewise_linear_interpolation_now = false;
	    if (fabs(relative_vox_sampling-1)>.01)	  
	      warning("because non-standard voxel size.\n");
	  }
      }
  }	    
}
// point should be w.r.t. middle of the scanner!
static inline 
void 
find_s_m_of_voxel(float& s, float& m,
		  const CartesianCoordinate3D<float>& point,
		  const float cphi, const float sphi,
		  const float tantheta)
{
  s = (point.x()*cphi+point.y()*sphi);

  m =
    point.z()- tantheta*(-point.x()*sphi+point.y()*cphi);
}


/*

interpolationkernel[s,a,b] b == interpolationkernel[s,b,a] a
normalised such that its integral == a

interpolationkernel[s_,ssize_,xsize_]  \
:=If[Abs[s]<Abs[ssize-xsize]/2,Min[1,ssize/
  xsize],If[Abs[s]>(ssize+xsize)/2,0,(-Abs[s]+(ssize+xsize)/2)/xsize]]
*/
// piecewise_linear interpolation for bin between -1 and 1
// vox
static inline
float 
piecewise_linear_interpolate(const float s, const float vox_size)
{
  if (fabs(s)<fabs(2-vox_size)/2)
    return std::min(1.F,2/vox_size);
  else if (fabs(s)>(2+vox_size)/2)
    return 0;
  else
    return (-fabs(s)+(2+vox_size)/2)/vox_size;
}

// linear interpolation between -1 and 1
static inline
float 
linear_interpolate(const float t)
{
  const float abst = fabs(t);
  if (abst>=1)
    return 0;
  else
    return 1-abst;
}

static inline
float 
interpolate_tang_pos(const float tang_pos_diff)
{
  return linear_interpolate(tang_pos_diff);
}


float
ProjMatrixByBinUsingInterpolation::
get_element(const Bin& bin, 
	    const CartesianCoordinate3D<float>& densel_ctr) const
{
  const float phi = proj_data_info_ptr->get_phi(bin);
  const float cphi = cos(phi);
  const float sphi = sin(phi);  
  const float tantheta = proj_data_info_ptr->get_tantheta(bin);

  float s_densel, m_densel;
  find_s_m_of_voxel(s_densel, m_densel,
		    densel_ctr,
		    cphi, sphi,
		    tantheta);
  const float s_diff =
    s_densel - proj_data_info_ptr->get_s(bin);

  const float m_diff =  
    m_densel -
    proj_data_info_ptr->get_m(bin); 

#if 0
  // alternative way to get m_diff using other code
  // should give the same, but doesn't.
  // for test case if scanner with 3 rings, span=1,
  // there is a difference of delta/2*proj_data_info_ptr->get_sampling_in_m(bin)
  // TODO why?

  // find correspondence between ax_pos coordinates and image coordinates:
  // z = num_planes_per_axial_pos * ring + axial_pos_to_z_offset
  const DataSymmetriesForBins_PET_CartesianGrid& symmetries =
    static_cast<const DataSymmetriesForBins_PET_CartesianGrid&>(
								*symmetries_ptr);
  assert(fabs(proj_data_info_ptr->get_sampling_in_m(bin) -
	      voxel_size.z() *
	      symmetries.get_num_planes_per_axial_pos(bin.segment_num()))
	 < 1.E-3);
  const float axial_pos_to_z_offset = 
    symmetries.get_axial_pos_to_z_offset(bin.segment_num());
  const float axial_pos_to_z_offset_in_mm = 
    axial_pos_to_z_offset * voxel_size.z();

  // note: origin.z() used because already in DataSymmetries
  const float other_m_diff =
    m - origin.z() - axial_pos_to_z_offset_in_mm
     - bin.axial_pos_num()*proj_data_info_ptr->get_sampling_in_m(bin);
  assert(fabs(m_diff - other_m_diff)
	 < .001*proj_data_info_ptr->get_sampling_in_m(bin));
#endif

  const float s_max =
    std::max(cphi>sphi? voxel_size.x() : voxel_size.y(),
	     proj_data_info_ptr->get_sampling_in_s(bin));
  float result =  interpolate_tang_pos(s_diff/s_max);
  if (result==0)
    return 0;
  const float m_max =
    std::max(voxel_size.z(),
	     proj_data_info_ptr->get_sampling_in_m(bin));

  result *=
    (use_piecewise_linear_interpolation_now?
     piecewise_linear_interpolate(m_diff/m_max, 
				  std::min(voxel_size.z(),
					   proj_data_info_ptr->get_sampling_in_m(bin))
				  /m_max)
     :
     linear_interpolate(m_diff/m_max)
     );

  if (result==0)
    return 0;

  return
    result *
    jacobian(proj_data_info_cyl().get_average_ring_difference(bin.segment_num()), 
	     proj_data_info_ptr->get_s(bin));
}
  
  
void 
ProjMatrixByBinUsingInterpolation::
calculate_proj_matrix_elems_for_one_bin(
                                        ProjMatrixElemsForOneBin& lor) const
{
  const Bin& bin = lor.get_bin();
  assert(bin.segment_num() >= proj_data_info_ptr->get_min_segment_num());    
  assert(bin.segment_num() <= proj_data_info_ptr->get_max_segment_num());    

  assert(lor.size() == 0);

  /* TODO
     there is a terrible problem with symmetries.
     We should not include densels that will be brought OUT-of-range
     using symmetries, unless it's checked (and we currently check only
     on the z-coordinate.
     However, we should include densels that will be brought IN-the-range
     using symmetries. Even if we use only basic_bins here.

     The last problem is illustrated by using axial_pos_num=0, which
     is usually at the edge of the image. So, we have to include voxels
     with negative z...

     Horrible.
  */
  BasicCoordinate<3,int> c;
  int min1;
  int max1;
  // find z-range (this would depend on origin and the symmetries though)
  {
#if 0
    /* attempt to take a large range of essentially 3 times the z-range of the image.
       This does not work though. It's easy to come up with cases where this
       range is still too small. Probably we would have to use the
       scanner length or so and take the image-origin into account.
       It's not easy though as illustrated by the following example: 

       2 rings, span=1, z-voxel_size=ring_distance/2, 3 image-planes, segment 0
       and image-origin shifted over 3 planes.
       The problem comes when using the swap_*_zq symmetries (which occurs
       even if shift_z=false).
    */
    min1=densel_range.get_min_index();
    max1=densel_range.get_max_index();
    int length = max-min1+1;
    min1 -= length;
    max1 += length;
#else
    /* Here we use geometric info. However, the code below only works
       for DiscretisedDensityOnCartesianGrid (with a regular_range)
    */
    min1=densel_range.get_min_index();
    // find radius of cylinder around all of the image
    const float max_radius = 
      std::max(
	       std::max(-densel_range[min1].get_min_index(),
			densel_range[min1].get_max_index()
			)*voxel_size[2],
	       std::max(-densel_range[min1][0].get_min_index(),
			densel_range[min1][0].get_max_index()
			)*voxel_size[1]
	       );
    // find width of the 'tube of response'
    const float z_width_of_TOR =
      proj_data_info_ptr->get_sampling_in_m(bin);
    // now find where tube enters/leaves image
    // we use a safety margin here as we could probably use z_width_of_LOR/2
    const float z_middle_LOR =
      proj_data_info_ptr->get_m(bin) - origin.z();
    const float min_z_LOR =
      z_middle_LOR - 
      fabs(proj_data_info_ptr->get_tantheta(bin))*max_radius -
      z_width_of_TOR;
    const float max_z_LOR =
      z_middle_LOR + 
      fabs(proj_data_info_ptr->get_tantheta(bin))*max_radius +
      z_width_of_TOR;

    min1 = round(floor(min_z_LOR/voxel_size[1]));
    max1 = round(ceil(max_z_LOR/voxel_size[1]));
#endif
  }
  /* we loop over all coordinates, but for optimisation do the following:
     In each dimension, we ASSUME that the non-zero range is CONNECTED.
     So, we keep track if a non-zero element was found and break
     out of the loop if we find a 0 after finding a non-zero.
  */
  bool found_nonzero1=false, found_nonzero2, found_nonzero3;
  for (c[1]=min1; c[1]<=max1; ++c[1])
    {
      // note: because c1 can be outside the allowed range, we have
      // in principle trouble getting the min_index() for c[2].
      // We'll assume that it is the same as for a(ny) c[1] within the range.
      // That's of course ok for VoxelsOnCartesianGrid
        const IndexRange<2>& range2d =
	  densel_range[std::min(std::max(c[1],densel_range.get_min_index()),
				densel_range.get_max_index())];

#if 0
      const int min2=range2d.get_min_index();
      const int max2=range2d.get_max_index();
#else
      // TODO ugly stuff to avoid having symmetries obtaining voxels
      // which are outside the FOV
      // this will break when non-zero origin.y() or x() 
      // (but then there would be no relevant symmetries I guess)
      assert(origin.y()==0);
      assert(origin.x()==0);
      const int first_min2=range2d.get_min_index();
      const int first_max2=range2d.get_max_index();
      const int min2 = std::max(first_min2, -first_max2);
      const int max2 = std::min(-first_min2, first_max2);
#endif
      found_nonzero2=false;
      for (c[2]=min2; c[2]<=max2; ++c[2])
	{
#if 0
	  const int min3=range2d[c[2]].get_min_index();
	  const int max3=range2d[c[2]].get_max_index();
#else
	  // TODO ugly stuff to avoid having symmetries obtaining voxels
	  // which are outside the FOV
	  // this will break when non-zero origin.y() or x() 
	  const int first_min3=range2d[c[2]].get_min_index();
	  const int first_max3=range2d[c[2]].get_max_index();
	  const int min3 = std::max(first_min3, -first_max3);
	  const int max3 = std::min(-first_min3, first_max3);
#endif
	  found_nonzero3 = false;
	  for (c[3]=min3; c[3]<=max3; ++c[3])
	    {
	      // TODO call a virtual function of DiscretisedDensity?
	      const CartesianCoordinate3D<float> coords = 
		CartesianCoordinate3D<float>(c[1]*voxel_size[1],
					     c[2]*voxel_size[2],
					     c[3]*voxel_size[3])
		+origin;
	      const float element_value =
		get_element(bin, coords);
	      if (element_value>0)
		{
		  found_nonzero3=true;
		  lor.push_back(ProjMatrixElemsForOneBin::value_type(c, element_value));
		}
#ifndef __PMByBinElement_SLOW__
	      else if (found_nonzero3)
		break;
#endif
	    }
	  if (found_nonzero3)
	    found_nonzero2=true;
#ifndef __PMByBinElement_SLOW__
	  else if (found_nonzero2)
	    break;
#endif
	}
      if (found_nonzero2)
	found_nonzero1=true;
#ifndef __PMByBinElement_SLOW__
      else if (found_nonzero1)
	break;
#endif
    }
}

END_NAMESPACE_STIR
