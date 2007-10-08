//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementations for non-inline functions of class stir::SegmentByView

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"

START_NAMESPACE_STIR


template <typename elemT>
SegmentByView<elemT>::
SegmentByView(const Array<3,elemT>& v,  
              const shared_ptr<ProjDataInfo>& pdi_ptr, 
              const int segment_num)
  :
  Segment<elemT>(pdi_ptr, segment_num), 
  Array<3,elemT>(v)
{
  assert( get_min_view_num() == pdi_ptr->get_min_view_num());
  assert( get_max_view_num() == pdi_ptr->get_max_view_num());
  assert( get_min_axial_pos_num() == pdi_ptr->get_min_axial_pos_num(segment_num));
  assert( get_max_axial_pos_num() == pdi_ptr->get_max_axial_pos_num(segment_num));
  assert( get_min_tangential_pos_num() == pdi_ptr->get_min_tangential_pos_num());
  assert( get_max_tangential_pos_num() == pdi_ptr->get_max_tangential_pos_num());

}

template <typename elemT>
SegmentByView<elemT>::
SegmentByView(const shared_ptr<ProjDataInfo>& pdi_ptr,
              const int segment_num)
  :
  Segment<elemT>(pdi_ptr, segment_num),
  Array<3,elemT>(IndexRange3D(pdi_ptr->get_min_view_num(),
                              pdi_ptr->get_max_view_num(),
		              pdi_ptr->get_min_axial_pos_num(segment_num),
                              pdi_ptr->get_max_axial_pos_num(segment_num),
		              pdi_ptr->get_min_tangential_pos_num(),
                              pdi_ptr->get_max_tangential_pos_num()))
{}

template <typename elemT>
SegmentByView<elemT>::SegmentByView(const SegmentBySinogram<elemT>& s_s)
  : Segment<elemT>(s_s.get_proj_data_info_ptr()->clone(),
            s_s.get_segment_num()), 
	       
    Array<3,elemT> (IndexRange3D(s_s.get_min_view_num(),s_s.get_max_view_num(),
		                 s_s.get_min_axial_pos_num(),s_s.get_max_axial_pos_num(),
		                 s_s.get_min_tangential_pos_num(), s_s.get_max_tangential_pos_num()))
{  
  
  for (int v=get_min_view_num(); v<= get_max_view_num(); v++)
    set_viewgram(s_s.get_viewgram(v));
}


template<typename elemT>
bool 
SegmentByView<elemT>::
operator ==(const Segment<elemT>& that) const
{
  return
    this->has_same_characteristics(that) &&
    Array<3,elemT>::operator==(static_cast<const self_type&>(that));
}

template <typename elemT>
Sinogram<elemT> 
SegmentByView<elemT>::get_sinogram(int axial_pos_num) const
{
  // gcc 2.95.2 needs a this-> in front of get_min_voew_num for unclear reasons
  Array<2,elemT> pre_sino(IndexRange2D(this->get_min_view_num(), get_max_view_num(), 
                                       get_min_tangential_pos_num(),get_max_tangential_pos_num()));
  for (int v=get_min_view_num(); v<= get_max_view_num(); v++)
    pre_sino[v] = Array<3,elemT>::operator[](v)[axial_pos_num];
  
  return Sinogram<elemT>(pre_sino, this->proj_data_info_ptr, axial_pos_num, 
			 this->get_segment_num());
}

template <typename elemT>
void
SegmentByView<elemT>::set_sinogram(const Sinogram<elemT>& sino, int axial_pos_num)
{
  for (int v=get_min_view_num(); v<= get_max_view_num(); v++)
    Array<3,elemT>::operator[](v)[axial_pos_num] = sino[v];
}




/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
SegmentByView<elemT>::
resize(const IndexRange<3>& range)
{   
  if (range == this->get_index_range())
    return;

  assert(range.is_regular()==true);

  // can only handle min_view==0 at the moment
  // TODO
  assert(range.get_min_index() == 0);

  ProjDataInfo* pdi_ptr = this->proj_data_info_ptr->clone();

  const int ax_min = range[0].get_min_index();
  const int ax_max = range[0].get_max_index();
  
  pdi_ptr->set_min_axial_pos_num(ax_min, this->get_segment_num());
  pdi_ptr->set_max_axial_pos_num(ax_max, this->get_segment_num());
  
  pdi_ptr->set_num_views(range.get_max_index() + 1);
  pdi_ptr->set_min_tangential_pos_num(range[0][ax_min].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[0][ax_min].get_max_index());

  this->proj_data_info_ptr = pdi_ptr;

  Array<3,elemT>::resize(range);
	
}

/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
SegmentByView<elemT>::
grow(const IndexRange<3>& range)
{
  resize(range);
}

/*************************************
 instantiations
 *************************************/

template class SegmentByView<float>;


END_NAMESPACE_STIR
