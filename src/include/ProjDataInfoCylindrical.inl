//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
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
}

float
ProjDataInfoCylindrical::get_phi(int segment_num,int view_num,int axial_position_num, int transaxial_position_num)const
{ return view_num*azimuthal_angle_sampling;}

float
ProjDataInfoCylindrical::get_t(int segment_num,int view_num,int axial_position_num, int transaxial_position_num) const
{return axial_position_num*get_axial_sampling(segment_num); }


#ifdef SET
void
ProjDataInfoCylindrical::set_azimuthal_angle_sampling(const float angle_v)
{azimuthal_angle_sampling =  angle_v;}
#endif
//void
//ProjDataInfoCylindrical::set_axial_sampling(const float samp_v, int segment_num)
//{axial_sampling = samp_v;}

float
ProjDataInfoCylindrical::get_azimuthal_angle_sampling() const
{return azimuthal_angle_sampling;}

float
ProjDataInfoCylindrical::get_axial_sampling(int segment_num) const
{  /*if (segment_num ==0)
    return axial_sampling_segment0;
  else
    return axial_sampling_other_segs;*/
  if (max_ring_diff[segment_num] != min_ring_diff[segment_num])
    return ring_spacing/2;
  else
    return ring_spacing;
}

float 
ProjDataInfoCylindrical::get_average_ring_difference(int segment_num) const
{
  return (min_ring_diff[segment_num] + max_ring_diff[segment_num])/2;
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


void
ProjDataInfoCylindrical::set_min_ring_difference( int min_ring_diff_v, int segment_num)
{
 min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_max_ring_difference( int max_ring_diff_v, int segment_num)
{
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_ring_spacing(float ring_spacing_v)
{
  ring_spacing = ring_spacing_v;
}




END_NAMESPACE_TOMO


  
