//
//
/*!

  \file
  \ingroup projdata

  \brief Implementations of inline functions of class stir::SegmentByView

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmtih Imanet Ltd
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

START_NAMESPACE_STIR

template <typename elemT>
int
SegmentByView<elemT>::get_num_views() const
{
  return this->get_length();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_min_view_num() const
{
  return this->get_min_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_max_view_num() const
{
  return this->get_max_index();
}

template <typename elemT>
int
SegmentByView<elemT>::get_num_axial_poss() const
 {
     return this->get_length()==0 ? 0 : (*this)[get_min_view_num()].get_length();
 }

template <typename elemT>
int
SegmentByView<elemT>::get_min_axial_pos_num() const
{
   return this->get_length()==0 ? 0 : (*this)[get_min_view_num()].get_min_index();
}

template <typename elemT>
int
SegmentByView<elemT>::get_max_axial_pos_num() const
{
 return this->get_length()==0 ? 0 : (*this)[get_min_view_num()].get_max_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_num_tangential_poss() const
{
  return this->get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_length();
}

template <typename elemT>
int
SegmentByView<elemT>::get_min_tangential_pos_num() const
{
return this->get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_min_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_max_tangential_pos_num()const
{
return this->get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_max_index();
}

template <typename elemT>
typename SegmentByView<elemT>::StorageOrder 
SegmentByView<elemT>::get_storage_order() const
{ return Segment<elemT>::StorageByView; }

template <typename elemT>
Viewgram<elemT> 
SegmentByView<elemT>::get_viewgram(int view_num) const
{ 
  return Viewgram<elemT>(Array<3,elemT>::operator[](view_num), 
			 this->proj_data_info_ptr->create_shared_clone(), view_num, 
			 this->get_segment_num()); }

template <typename elemT>
void 
SegmentByView<elemT>::set_sinogram(const Sinogram<elemT> &s)
{ set_sinogram(s, s.get_axial_pos_num()); }

template <typename elemT>
void 
SegmentByView<elemT>::set_viewgram(const Viewgram<elemT> &v/*, int view_num*/)
{ Array<3,elemT>::operator[](v.get_view_num()) = v; }

END_NAMESPACE_STIR
