//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007,Hammersmith Imanet Ltd
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

  \brief Implementations of inline functions of class stir::Sinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR


template <typename elemT>
int
Sinogram<elemT>::get_segment_num() const
{ return segment_num; }

template <typename elemT>
int
Sinogram<elemT>::get_axial_pos_num() const
{ return axial_pos_num; }

template <typename elemT>
int
Sinogram<elemT>::get_timing_pos_num() const
{ return timing_pos_num; }

template <typename elemT>
int
Sinogram<elemT>::get_min_view_num() const
  {return this->get_min_index();}

template <typename elemT>
int
Sinogram<elemT>::get_max_view_num() const
  { return this->get_max_index(); }

template <typename elemT>
int
Sinogram<elemT>::get_num_views() const
  { return this->get_length();}

template <typename elemT>
int
Sinogram<elemT>::get_num_tangential_poss()const
  { return this->get_length()==0 ? 0 : (*this)[get_min_view_num()].get_length();}

template <typename elemT>
int
Sinogram<elemT>::get_min_tangential_pos_num() const
  { return this->get_length()==0 ? 0 :(*this)[get_min_view_num()].get_min_index();}

template <typename elemT>
int
Sinogram<elemT>::get_max_tangential_pos_num() const
{ return this->get_length()==0 ? 0 :(*this)[get_min_view_num()].get_max_index(); }



template <typename elemT>
Sinogram<elemT>
Sinogram<elemT>::get_empty_copy(void) const
{
    Sinogram<elemT> copy(proj_data_info_ptr, get_axial_pos_num(), get_segment_num(), get_timing_pos_num());
    return copy;
}


template <typename elemT>
const ProjDataInfo*
Sinogram<elemT>:: get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}

template <typename elemT>
shared_ptr<ProjDataInfo>
Sinogram<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_ptr;
}

template <typename elemT>
Sinogram<elemT>::
Sinogram(const Array<2,elemT>& p, 
         const shared_ptr<ProjDataInfo >& pdi_ptr, 
         const int ax_pos_num, const int s_num, const int t_num) 
  :
  Array<2,elemT>(p), 
  proj_data_info_ptr(pdi_ptr),
  axial_pos_num(ax_pos_num), 
  segment_num(s_num),
  timing_pos_num(t_num)
{
  assert(axial_pos_num <= proj_data_info_ptr->get_max_axial_pos_num(segment_num));
  assert(axial_pos_num >= proj_data_info_ptr->get_min_axial_pos_num(segment_num));
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)

  assert( get_min_view_num() == pdi_ptr->get_min_view_num());
  assert( get_max_view_num() == pdi_ptr->get_max_view_num());
  assert( get_min_tangential_pos_num() == pdi_ptr->get_min_tangential_pos_num());
  assert( get_max_tangential_pos_num() == pdi_ptr->get_max_tangential_pos_num());
}



template <typename elemT>
Sinogram<elemT>::
Sinogram(const shared_ptr<ProjDataInfo >& pdi_ptr, 
         const int ax_pos_num, const int s_num, const int t_num) 
  :
  Array<2,elemT>(IndexRange2D (pdi_ptr->get_min_view_num(),
			       pdi_ptr->get_max_view_num(),
			       pdi_ptr->get_min_tangential_pos_num(),
			       pdi_ptr->get_max_tangential_pos_num())), 
  proj_data_info_ptr(pdi_ptr),
  axial_pos_num(ax_pos_num),
  segment_num(s_num),
  timing_pos_num(t_num)
{
  assert(axial_pos_num <= proj_data_info_ptr->get_max_axial_pos_num(segment_num));
  assert(axial_pos_num >= proj_data_info_ptr->get_min_axial_pos_num(segment_num));
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)
}


END_NAMESPACE_STIR
