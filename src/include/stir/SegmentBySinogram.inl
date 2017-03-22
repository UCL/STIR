//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
                         Segment<elemT>::proj_data_info_ptr, axial_pos_num, 
                         Segment<elemT>::get_segment_num(),
						 Segment<elemT>::get_timing_pos_num()); }

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
