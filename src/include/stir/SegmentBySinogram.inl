//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations of inline functions of class SegmentBySinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
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

template <typename elemT>
int 
SegmentBySinogram<elemT> ::get_num_axial_poss() const
{
   return get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_axial_pos_num() const
{
  return get_min_index();
}

template <typename elemT>
int 
SegmentBySinogram<elemT>::get_max_axial_pos_num() const
{
  return get_max_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_num_views() const
{
  return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_view_num() const
{
  return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_min_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_max_view_num() const
{
return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_max_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_num_tangential_poss() const
{
  return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_tangential_pos_num() const
{
 return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_min_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_max_tangential_pos_num() const
{
return get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_max_index();
}
 
template <typename elemT>
SegmentBySinogram<elemT>::StorageOrder 
SegmentBySinogram<elemT>::
get_storage_order() const
  { return StorageBySino; }

template <typename elemT>
Sinogram<elemT> 
SegmentBySinogram<elemT>::
get_sinogram(int axial_pos_num) const
{ return Sinogram<elemT>(Array<3,elemT>::operator[](axial_pos_num), 
                         proj_data_info_ptr, axial_pos_num, 
                         get_segment_num()); }

template <typename elemT>
void 
SegmentBySinogram<elemT>::
set_sinogram(Sinogram<elemT> const &s, int axial_pos_num)
{ Array<3,elemT>::operator[](axial_pos_num) = s; }

template <typename elemT>
void 
SegmentBySinogram<elemT>::
set_sinogram(const Sinogram<elemT>& s)
  { set_sinogram(s, s.get_axial_pos_num()); }

END_NAMESPACE_STIR
