//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Non-inline implementations of ProjDataInfoCylindrical

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "ProjDataInfoCylindrical.h"
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif
#if 0
#include <iostream>
#endif

#ifndef TOMO_NO_NAMESPACES
using std::min_element;
using std::max_element;
using std::swap;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_TOMO

ProjDataInfoCylindrical::
ProjDataInfoCylindrical()
{}


ProjDataInfoCylindrical::
ProjDataInfoCylindrical(const shared_ptr<Scanner>& scanner_ptr,
                        const VectorWithOffset<int>& num_axial_pos_per_segment,
                        const VectorWithOffset<int>& min_ring_diff_v, 
                        const VectorWithOffset<int>& max_ring_diff_v,
                        const int num_views,const int num_tangential_poss)
  :ProjDataInfo(scanner_ptr,num_axial_pos_per_segment, 
                num_views,num_tangential_poss),
   min_ring_diff(min_ring_diff_v),
   max_ring_diff(max_ring_diff_v)
{
  
  azimuthal_angle_sampling = _PI/num_views;
  ring_radius = get_scanner_ptr()->get_ring_radius();
  ring_spacing= get_scanner_ptr()->get_ring_spacing() ;

  view_mashing_factor = get_scanner_ptr()->get_num_detectors_per_ring()/2 / num_views;
  assert(get_scanner_ptr()->get_num_detectors_per_ring() % (2*num_views) == 0);

  assert(min_ring_diff.get_length() == max_ring_diff.get_length());
  assert(min_ring_diff.get_length() == num_axial_pos_per_segment.get_length());

  // check min,max ring diff
  {
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num]> max_ring_diff[segment_num])
      {
        warning("ProjDataInfoCylindrical: min_ring_difference %d is larger than max_ring_difference %d for segment %d. "
          "Swapping them around\n", 
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);
        swap(min_ring_diff[segment_num], max_ring_diff[segment_num]);
      }
  }

  initialise_ring_diff_arrays();
}

void
ProjDataInfoCylindrical::
initialise_ring_diff_arrays() const
{

  // check min,max ring diff
  {
    // check is necessary here again because of set_min_ring_difference()
    // we do not swap here because that would require the min/max_ring_diff arrays to be mutable as well
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num]> max_ring_diff[segment_num])
      {
        error("ProjDataInfoCylindrical: min_ring_difference %d is larger than max_ring_difference %d for segment %d.\n",
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);        
      }
  }
  // initialise m_offset 
  { 
    m_offset.grow(get_min_segment_num(),get_max_segment_num());
    
    /* m_offsets are found by requiring
    get_m(..., min_axial_pos_num,...) == - get_m(..., max_axial_pos_num,...)
    */
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
    {
      m_offset[segment_num] =
        ((get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num))
        *get_axial_sampling(segment_num)
        )/2;
    }
  }
  // initialise ax_pos_num_offset 
  { 
    const int num_rings = get_scanner_ptr()->get_num_rings();
    ax_pos_num_offset.grow(get_min_segment_num(),get_max_segment_num());
    
    /* ax_pos_num will be determined by looking at ring1+ring2
       it's a bit complicated because of dependency on the 'axial compression' (or 'span')
       If get_num_rings_per_axial_pos(segment_num)==1 (i.e. no axial compression)
       then ring1+ring2 increments in steps of 2 in the segment. So, 
       ax_pos_num = (ring1+ring2)/2 + some offset.
       In the other case (i.e. axial compression), ring1+ring2 increments in steps of 1, so
       ax_pos_num = (ring1+ring2) + some offset.
       */
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
    {
#if 0 
      // TODO all this wouldn't work on HiDAC data or so (for instance even number of planes in seg 0)
      // this first offset would shift the 0 of ax_pos_num to the centre of the scanner
      ax_pos_num_offset[segment_num] =
        num_rings - 1;
      
      // now shift origin such that the middle of the ax_pos_num range corresponds to the the centre of the scanner
      // first check that we don't end up with half-integer indices
      assert((get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num)) %
          get_num_rings_per_axial_pos(segment_num) == 0);
#endif
      ax_pos_num_offset[segment_num] -=
          (get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num))/
          get_num_rings_per_axial_pos(segment_num);
    }
  }
  // initialise ring_diff_to_segment_num
  {
    const int min_ring_difference = 
      *min_element(min_ring_diff.begin(), min_ring_diff.end());
    const int max_ring_difference = 
      *max_element(max_ring_diff.begin(), max_ring_diff.end());
    ring_diff_to_segment_num.grow(min_ring_difference, max_ring_difference);

    for(int ring_diff=min_ring_difference; ring_diff <= max_ring_difference; ++ring_diff) 
    {    
      int segment_num;
      for (segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      {
        if (ring_diff >= min_ring_diff[segment_num] &&
            ring_diff <= max_ring_diff[segment_num])
        {
#if 0
          std::cerr << "ring diff " << ring_diff << " stored in s:" << segment_num << std::endl;
#endif
          ring_diff_to_segment_num[ring_diff] = segment_num;
          break;
        }
      }
      if (segment_num>get_max_segment_num())
      {
        warning("ProjDataInfoCylindrical: ring difference %d does not belong to a segment\n",
          ring_diff);
        ring_diff_to_segment_num[ring_diff] = get_max_segment_num()+1;
      }
    }
  }

  ring_diff_arrays_computed = true;
}


/*
void
ProjDataInfoCylindrical::
set_azimuthal_angle_sampling(const float angle_v)
{azimuthal_angle_sampling =  angle_v;}

void
ProjDataInfoCylindrical::
set_axial_sampling(const float samp_v, int segment_num)
{axial_sampling = samp_v;}
*/

void
ProjDataInfoCylindrical::
set_min_ring_difference( int min_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoCylindrical::
set_max_ring_difference( int max_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoCylindrical::
set_ring_spacing(float ring_spacing_v)
{
  ring_spacing = ring_spacing_v;
}


string
ProjDataInfoCylindrical::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif  
  s << ProjDataInfo::parameter_info();

  s << "ring differences per segment: \n";
  for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
  {
    s << '(' << min_ring_diff[segment_num]  << ',' << max_ring_diff[segment_num] <<')';
  }
  s << "\n";
  s << ends;
  return s.str();
}

END_NAMESPACE_TOMO
