//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \brief Implementations for non-inline functions of class stir::SegmentBySinogram

  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"

START_NAMESPACE_STIR



template <typename elemT>
SegmentBySinogram<elemT> ::
SegmentBySinogram(const Array<3,elemT>& v, 
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
SegmentBySinogram<elemT> ::
SegmentBySinogram(const shared_ptr<ProjDataInfo>& pdi_ptr,
		  const int segment_num)
  : 
  Segment<elemT>(pdi_ptr, segment_num), 
  Array<3,elemT>(IndexRange3D(pdi_ptr->get_min_axial_pos_num(segment_num),
                              pdi_ptr->get_max_axial_pos_num(segment_num),
                              pdi_ptr->get_min_view_num(),
                              pdi_ptr->get_max_view_num(),
                              pdi_ptr->get_min_tangential_pos_num(),
                              pdi_ptr->get_max_tangential_pos_num()))
{}

template <typename elemT>
SegmentBySinogram<elemT>::
SegmentBySinogram(const SegmentByView<elemT>& s_v )
  : Segment<elemT>(s_v.get_proj_data_info_ptr()->create_shared_clone(),
                   s_v.get_segment_num()),	      
   Array<3,elemT> (IndexRange3D (s_v.get_min_axial_pos_num(), s_v.get_max_axial_pos_num(),
		                 s_v.get_min_view_num(), s_v.get_max_view_num(),
		                 s_v.get_min_tangential_pos_num(), s_v.get_max_tangential_pos_num()))
{
  
  for (int r=get_min_axial_pos_num(); r<= get_max_axial_pos_num(); r++)
    set_sinogram(s_v.get_sinogram(r));
}

template<typename elemT>
bool 
SegmentBySinogram<elemT>::
operator ==(const Segment<elemT>& that) const
{
  return
    this->has_same_characteristics(that) &&
    Array<3,elemT>::operator==(static_cast<const self_type&>(that));
}


template <typename elemT>
Viewgram<elemT> 
SegmentBySinogram<elemT>::get_viewgram(int view_num) const
{
  // gcc 2.95.2 needs a this-> in front of get_min_ring for unclear reasons
  Array<2,elemT> pre_view(IndexRange2D(this->get_min_axial_pos_num(), get_max_axial_pos_num(),
                                       get_min_tangential_pos_num(),get_max_tangential_pos_num()));
  for (int r=get_min_axial_pos_num(); r<= get_max_axial_pos_num(); r++)
    pre_view[r] = Array<3,elemT>::operator[](r)[view_num];
  //KT 9/12 constructed a PETSinogram before...
  // CL&KT 15/12 added ring_difference stuff
  return Viewgram<elemT>(pre_view, this->proj_data_info_ptr->create_shared_clone(), view_num, 
			 this->get_segment_num());
}

template <typename elemT>
void
SegmentBySinogram<elemT>::set_viewgram(const Viewgram<elemT>& viewgram)
{
  for (int r=get_min_axial_pos_num(); r<= get_max_axial_pos_num(); r++)
    Array<3,elemT>::operator[](r)[viewgram.get_view_num()] = viewgram[r];
}



/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
SegmentBySinogram<elemT>::
resize(const IndexRange<3>& range)
{   
  if (range == this->get_index_range())
    return;

  assert(range.is_regular()==true);

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();

  // can only handle min_view==0 at the moment
  // TODO
  assert(range[ax_min].get_min_index() == 0);

  shared_ptr<ProjDataInfo> pdi_ptr = this->proj_data_info_ptr->create_shared_clone();
  
  pdi_ptr->set_min_axial_pos_num(ax_min, this->get_segment_num());
  pdi_ptr->set_max_axial_pos_num(ax_max, this->get_segment_num());
  
  pdi_ptr->set_num_views(range[ax_min].get_max_index() + 1);
  pdi_ptr->set_min_tangential_pos_num(range[ax_min][0].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[ax_min][0].get_max_index());

  this->proj_data_info_ptr = pdi_ptr;

  Array<3,elemT>::resize(range);
	
}


/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
SegmentBySinogram<elemT>::
grow(const IndexRange<3>& range)
{
  resize(range);
}

/*************************************
 instantiations
 *************************************/

template class SegmentBySinogram<float>;

END_NAMESPACE_STIR
