//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations of inline functions for class ProjDataInfo

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

START_NAMESPACE_TOMO


void 
ProjDataInfo::set_num_views(const int num_views)
{
  min_view_num = 0;
  max_view_num = num_views-1;
}

void 
ProjDataInfo::set_num_tangential_poss(const int num_tang_poss)
{

  min_tangential_pos_num = -(num_tang_poss/2);
  max_tangential_pos_num = min_tangential_pos_num + num_tang_poss-1;
}

void 
ProjDataInfo::set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_pos_per_segment)
{
  // first do assignments to make the members the correct size 
  // (data will be overwritten)
  min_axial_pos_per_seg= num_axial_pos_per_segment;
  max_axial_pos_per_seg= num_axial_pos_per_segment;
  
  for (int i=num_axial_pos_per_segment.get_min_index(); 
       i<=num_axial_pos_per_segment.get_max_index();
       i++)
  {
    min_axial_pos_per_seg[i]=0;
    max_axial_pos_per_seg[i]=num_axial_pos_per_segment[i]-1;
  }
    
}

void
ProjDataInfo::set_min_tangential_pos_num(int min_tang_poss)
{
  min_tangential_pos_num = min_tang_poss;
}

void
ProjDataInfo::set_max_tangential_pos_num(int max_tang_poss)
{
  max_tangential_pos_num = max_tang_poss;
}


ProjDataInfo::ProjDataInfo()
{}




ProjDataInfo::ProjDataInfo(const shared_ptr<Scanner> scanner_ptr_v,
                           const VectorWithOffset<int>& num_axial_pos_per_segment_v, 
                           const int num_views_v, 
                           const int num_tangential_poss_v)
			   :scanner_ptr(scanner_ptr_v)

{ 
  set_num_views(num_views_v);
  set_num_tangential_poss(num_tangential_poss_v);
  set_num_axial_poss_per_segment(num_axial_pos_per_segment_v);
}
  

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


END_NAMESPACE_TOMO

