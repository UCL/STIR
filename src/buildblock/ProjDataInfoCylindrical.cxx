//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Non-inline implementations of ProjDataInfoCylindrical

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "ProjDataInfoCylindrical.h"

START_NAMESPACE_TOMO

ProjDataInfoCylindrical::ProjDataInfoCylindrical()
{}


ProjDataInfoCylindrical::ProjDataInfoCylindrical(const shared_ptr<Scanner> scanner_ptr,
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
  assert(min_ring_diff.get_length() == max_ring_diff.get_length());
  assert(min_ring_diff.get_length() == num_axial_pos_per_segment.get_length());

  m_offset.grow(get_min_segment_num(),get_max_segment_num());

  /* m_offsets are found by requiring
    get_m(..., min_axial_pos_num,...) == - get_m(..., max_axial_pos_num,...)
  */
  for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
  {
    m_offset[segment_num] =
     ((get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num))
        *get_axial_sampling(segment_num)
       //+ get_ring_spacing()*get_average_ring_difference(segment_num) 
      )/2;
  }
}
 
END_NAMESPACE_TOMO
