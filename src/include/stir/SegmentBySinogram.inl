//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Implementations of inline functions of class stir::SegmentBySinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project


*/

START_NAMESPACE_STIR

template <typename elemT>
int 
SegmentBySinogram<elemT> ::get_num_axial_poss() const
{
   return this->get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_axial_pos_num() const
{
  return this->get_min_index();
}

template <typename elemT>
int 
SegmentBySinogram<elemT>::get_max_axial_pos_num() const
{
  return this->get_max_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_num_views() const
{
  return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_view_num() const
{
  return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_min_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_max_view_num() const
{
return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_max_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_num_tangential_poss() const
{
  return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_length();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_min_tangential_pos_num() const
{
 return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_min_index();
}

template <typename elemT>
int
SegmentBySinogram<elemT>::get_max_tangential_pos_num() const
{
return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()][get_min_view_num()].get_max_index();
}
 
template <typename elemT>
typename SegmentBySinogram<elemT>::StorageOrder 
SegmentBySinogram<elemT>::
get_storage_order() const
  { return Segment<elemT>::StorageBySino; }

template <typename elemT>
Sinogram<elemT> 
SegmentBySinogram<elemT>::
get_sinogram(int axial_pos_num) const
{ return Sinogram<elemT>(Array<3,elemT>::operator[](axial_pos_num), 
                         Segment<elemT>::proj_data_info_sptr, axial_pos_num,
                         Segment<elemT>::get_segment_num()); }

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
