//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Implementations of inline functions for class ProjDataInfo

  \author Sanida Mustafovic
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

START_NAMESPACE_STIR
  
bool 
ProjDataInfo::operator !=(const ProjDataInfo& proj) const
{ return !(*this == proj); }

int 
ProjDataInfo::get_num_segments() const
{ return (max_axial_pos_per_seg.get_length());}


int
ProjDataInfo::get_num_axial_poss(const int segment_num) const
{ return  max_axial_pos_per_seg[segment_num] - min_axial_pos_per_seg[segment_num]+1;}

int 
ProjDataInfo::get_num_views() const
{ return max_view_num - min_view_num + 1; }

int 
ProjDataInfo::get_num_tangential_poss() const
{ return  max_tangential_pos_num - min_tangential_pos_num + 1; }

int 
ProjDataInfo::get_min_segment_num() const
{ return (max_axial_pos_per_seg.get_min_index()); }

int 
ProjDataInfo::get_max_segment_num()const
{ return (max_axial_pos_per_seg.get_max_index());  }

int
ProjDataInfo::get_min_axial_pos_num(const int segment_num) const
{ return min_axial_pos_per_seg[segment_num];}


int
ProjDataInfo::get_max_axial_pos_num(const int segment_num) const
{ return max_axial_pos_per_seg[segment_num];}


int 
ProjDataInfo::get_min_view_num() const
  { return min_view_num; }

int 
ProjDataInfo::get_max_view_num()  const
{ return max_view_num; }


int 
ProjDataInfo::get_min_tangential_pos_num()const
{ return min_tangential_pos_num; }

int 
ProjDataInfo::get_max_tangential_pos_num()const
{ return max_tangential_pos_num; }



const 
Scanner*
ProjDataInfo::get_scanner_ptr() const
{ 
  return scanner_ptr.get();
    
}


END_NAMESPACE_STIR

