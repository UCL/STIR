//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

// for sqrt
#include <math.h>
#include "Bin.h"

START_NAMESPACE_TOMO


float
ProjDataInfoCylindrical::get_phi(const Bin& bin)const
{ return bin.view_num()*azimuthal_angle_sampling;}


float
ProjDataInfoCylindrical::get_m(const Bin& bin) const
{ 
  if (!ring_diff_arrays_computed)
    initialise_ring_diff_arrays();
  return 
    bin.axial_pos_num()*get_axial_sampling(bin.segment_num())
    - m_offset[bin.segment_num()];
}

float
ProjDataInfoCylindrical::get_t(const Bin& bin) const
{
  return 
    get_m(bin)/
    sqrt(1+square(get_tantheta(bin)));
}


float
ProjDataInfoCylindrical::get_tantheta(const Bin& bin) const
{
  return
    get_average_ring_difference(bin.segment_num())*
    ring_spacing/ 
    (2*sqrt(square(ring_radius)-square(get_s(bin))));  
}


int 
ProjDataInfoCylindrical::
get_num_rings_per_axial_pos(const int segment_num) const
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
  return ring_spacing/get_num_rings_per_axial_pos(segment_num);
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
{return ring_radius;}

float
ProjDataInfoCylindrical::get_ring_spacing() const
{ return ring_spacing;}

int 
ProjDataInfoCylindrical::
get_segment_num_for_ring_difference(const int ring_diff) const
{
  if (!ring_diff_arrays_computed)
    initialise_ring_diff_arrays();
  return ring_diff_to_segment_num[ring_diff];
}

int
ProjDataInfoCylindrical::
get_view_mashing_factor() const
{
  return view_mashing_factor;
}

void
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

  segment_num = get_segment_num_for_ring_difference(ring1-ring2);
  // see initialise_ring_diff_arrays() for some info
  if (get_num_rings_per_axial_pos(segment_num)==1)
    ax_pos_num = (ring1 + ring2 - ax_pos_num_offset[segment_num])/2;
  else
    ax_pos_num = (ring1 + ring2 - ax_pos_num_offset[segment_num]);
}

END_NAMESPACE_TOMO


  
