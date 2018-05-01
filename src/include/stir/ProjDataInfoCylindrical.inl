/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2003, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2018, University of Leeds

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
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Palak Wadhwa
  \author Berta Marti Fuster
  \author PARAPET project
*/

// for sqrt
#include <math.h>
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <algorithm>

START_NAMESPACE_STIR

void 
ProjDataInfoCylindrical::
initialise_ring_diff_arrays_if_not_done_yet() const
{
  // for efficiency reasons, use "Double-Checked-Locking(DCL) pattern" with OpenMP atomic operation
  // OpenMP v3.1 or later required
  // thanks to yohjp: http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = ring_diff_arrays_computed;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALRINGDIFFARRAY)
#endif
      {
        if (!ring_diff_arrays_computed)
          initialise_ring_diff_arrays();
      }
    }
}

//PW Added the view offset from the scanner, code may now support intrinsic tilt.
float
ProjDataInfoCylindrical::get_phi(const Bin& bin)const
{ return bin.view_num()*azimuthal_angle_sampling + scanner_ptr->get_default_intrinsic_tilt();}


float
ProjDataInfoCylindrical::get_m(const Bin& bin) const
{ 

  this->initialise_ring_diff_arrays_if_not_done_yet();
  return 
    bin.axial_pos_num()*get_axial_sampling(bin.segment_num())
    - m_offset[bin.segment_num()];
}

float
ProjDataInfoCylindrical::get_t(const Bin& bin) const
{
  return 
    get_m(bin)*get_costheta(bin);
}

float
ProjDataInfoCylindrical::get_tantheta(const Bin& bin) const
{
  const float delta=get_average_ring_difference(bin.segment_num());
  if (fabs(delta)<0.0001F)
    return 0;
  const float R=get_ring_radius(bin.view_num());
  assert(R>=fabs(get_s(bin)));
  return delta*ring_spacing/(2*sqrt(square(R)-square(get_s(bin))));
}


float 
ProjDataInfoCylindrical::get_sampling_in_m(const Bin& bin) const
{
  return get_axial_sampling(bin.segment_num());
}

float 
ProjDataInfoCylindrical::get_sampling_in_t(const Bin& bin) const
{
  return get_axial_sampling(bin.segment_num())*get_costheta(bin);
}

int 
ProjDataInfoCylindrical::
get_num_axial_poss_per_ring_inc(const int segment_num) const
{
  return
    max_ring_diff[segment_num] != min_ring_diff[segment_num] ?
    2 : 1;
}

float
ProjDataInfoCylindrical::get_azimuthal_angle_sampling() const
{return azimuthal_angle_sampling;}

float
ProjDataInfoCylindrical::get_axial_sampling(int segment_num) const
{
  return ring_spacing/get_num_axial_poss_per_ring_inc(segment_num);
}

float 
ProjDataInfoCylindrical::get_average_ring_difference(int segment_num) const
{
  // KT 05/07/2001 use float division here. 
  // In any reasonable case, min+max_ring_diff will be even.
  // But some day, an unreasonable case will walk in.
  return (min_ring_diff[segment_num] + max_ring_diff[segment_num])/2.F;
}


int 
ProjDataInfoCylindrical::get_min_ring_difference(int segment_num) const
{ return min_ring_diff[segment_num]; }

int 
ProjDataInfoCylindrical::get_max_ring_difference(int segment_num) const
{ return max_ring_diff[segment_num]; }

float
ProjDataInfoCylindrical::get_ring_radius() const
{
  if (this->ring_radius.get_min_index()!=0 || this->ring_radius.get_max_index()!=0)
    {
      // check if all elements are equal
      for (VectorWithOffset<float>::const_iterator iter=this->ring_radius.begin(); iter!= this->ring_radius.end(); ++iter)
	{
	  if (*iter != *this->ring_radius.begin())
	    error("get_ring_radius called for non-circular ring");
	}
    }
  return *this->ring_radius.begin();
}

void
ProjDataInfoCylindrical::set_ring_radii_for_all_views(const VectorWithOffset<float>& new_ring_radius)
{
  if (new_ring_radius.get_min_index() != this->get_min_view_num() ||
      new_ring_radius.get_max_index() != this->get_max_view_num())
    {
      error("error set_ring_radii_for_all_views: you need to use correct range of view numbers");
    }

  this->ring_radius = new_ring_radius;
}

VectorWithOffset<float>
ProjDataInfoCylindrical::get_ring_radii_for_all_views() const
{
  if (this->ring_radius.get_min_index()==0 && this->ring_radius.get_max_index()==0)
    {
      VectorWithOffset<float> out(this->get_min_view_num(), this->get_max_view_num());
      out.fill(this->ring_radius[0]);
      return out;
    }
  else
    return this->ring_radius;
}

float
ProjDataInfoCylindrical::get_ring_radius( const int view_num) const
{
  if (this->ring_radius.get_min_index()==0 && this->ring_radius.get_max_index()==0)
    return ring_radius[0];
  else
    return ring_radius[view_num];
}

float
ProjDataInfoCylindrical::get_ring_spacing() const
{ return ring_spacing;}

int
ProjDataInfoCylindrical::
get_view_mashing_factor() const
{
  // KT 10/05/2002 new assert
  assert(get_scanner_ptr()->get_num_detectors_per_ring() > 0);
  // KT 10/05/2002 moved assert here from constructor
  assert(get_scanner_ptr()->get_num_detectors_per_ring() % (2*get_num_views()) == 0);
  // KT 28/11/2001 do not pre-store anymore as set_num_views would invalidate it
  return get_scanner_ptr()->get_num_detectors_per_ring()/2 / get_num_views();
}

Succeeded
ProjDataInfoCylindrical::
get_segment_num_for_ring_difference(int& segment_num, const int ring_diff) const
{
  if (!sampling_corresponds_to_physical_rings)
    return Succeeded::no;

  // check currently necessary as reduce_segment does not reduce the size of the ring_diff arrays
  if (ring_diff > get_max_ring_difference(get_max_segment_num()) ||
      ring_diff < get_min_ring_difference(get_min_segment_num()))
    return Succeeded::no;

  this->initialise_ring_diff_arrays_if_not_done_yet();

  segment_num = ring_diff_to_segment_num[ring_diff];
  // warning: relies on initialise_ring_diff_arrays to set invalid ring_diff to a too large segment_num
  if (segment_num <= get_max_segment_num())
    return Succeeded::yes;
  else
    return Succeeded::no;
}


Succeeded
ProjDataInfoCylindrical::
get_segment_axial_pos_num_for_ring_pair(int& segment_num,
                                        int& ax_pos_num,
                                        const int ring1,
                                        const int ring2) const
{
  assert(0<=ring1);
  assert(ring1<get_scanner_ptr()->get_num_rings());
  assert(0<=ring2);
  assert(ring2<get_scanner_ptr()->get_num_rings());

  // KT 01/08/2002 swapped rings
  if (get_segment_num_for_ring_difference(segment_num, ring2-ring1) == Succeeded::no)
    return Succeeded::no;

  // see initialise_ring_diff_arrays() for some info
  ax_pos_num = (ring1 + ring2 - ax_pos_num_offset[segment_num])*
               get_num_axial_poss_per_ring_inc(segment_num)/2;
  return Succeeded::yes;
}

const ProjDataInfoCylindrical::RingNumPairs&
ProjDataInfoCylindrical::
get_all_ring_pairs_for_segment_axial_pos_num(const int segment_num,
					     const int axial_pos_num) const
{
  this->initialise_ring_diff_arrays_if_not_done_yet();
  if (is_null_ptr(segment_axial_pos_to_ring_pair[segment_num][axial_pos_num]))
    compute_segment_axial_pos_to_ring_pair(segment_num, axial_pos_num);
  return *segment_axial_pos_to_ring_pair[segment_num][axial_pos_num];
}

unsigned
ProjDataInfoCylindrical::
get_num_ring_pairs_for_segment_axial_pos_num(const int segment_num,
					     const int axial_pos_num) const
{
  return 
    static_cast<unsigned>(
       this->get_all_ring_pairs_for_segment_axial_pos_num(segment_num,axial_pos_num).size());
}

END_NAMESPACE_STIR


  
