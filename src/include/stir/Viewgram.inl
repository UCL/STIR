//
//
/*!

  \file
  \ingroup projdata

  \brief Inline implementations of class stir::Viewgram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

template <typename elemT>
ViewgramIndices
Viewgram<elemT>::get_viewgram_indices() const
{
  return this->_indices;
}

template <typename elemT>
int
Viewgram<elemT>::get_segment_num() const
{ return this->_indices.segment_num(); }

template <typename elemT>
int
Viewgram<elemT>::get_view_num() const
{ return this->_indices.view_num(); }

template <typename elemT>
int
Viewgram<elemT>::get_min_axial_pos_num() const
  {return this->get_min_index();}

template <typename elemT>
int
Viewgram<elemT>::get_max_axial_pos_num() const
  { return this->get_max_index(); }

template <typename elemT>
int
Viewgram<elemT>::get_num_axial_poss() const
  { return this->get_length();}


template <typename elemT>
int
Viewgram<elemT>::get_num_tangential_poss() const
  { return this->get_length()==0 ? 0 : (*this)[get_min_axial_pos_num()].get_length();}


template <typename elemT>
int
Viewgram<elemT>::get_min_tangential_pos_num() const
  { return this->get_length()==0 ? 0 :(*this)[get_min_axial_pos_num()].get_min_index();}

template <typename elemT>
int
Viewgram<elemT>::get_max_tangential_pos_num() const
{ return this->get_length()==0 ? 0 :(*this)[get_min_axial_pos_num()].get_max_index(); }


template <typename elemT>
Viewgram<elemT>
Viewgram<elemT>::get_empty_copy(void) const
  {
    Viewgram<elemT> copy(proj_data_info_sptr, get_viewgram_indices());
    return copy;
}

template <typename elemT>
shared_ptr<const ProjDataInfo>
Viewgram<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}


template <typename elemT>
Viewgram<elemT>::
Viewgram(const Array<2,elemT>& p, 
	 const shared_ptr<const ProjDataInfo>& pdi_sptr,
         const ViewgramIndices& ind)
  :
  Array<2,elemT>(p), proj_data_info_sptr(pdi_sptr),
  _indices(ind)
{
  assert(view_num <= proj_data_info_sptr->get_max_view_num());
  assert(view_num >= proj_data_info_sptr->get_min_view_num());
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)

  assert( get_min_axial_pos_num() == pdi_sptr->get_min_axial_pos_num(ind.segment_num()));
  assert( get_max_axial_pos_num() == pdi_sptr->get_max_axial_pos_num(ind.segment_num()));
  assert( get_min_tangential_pos_num() == pdi_sptr->get_min_tangential_pos_num());
  assert( get_max_tangential_pos_num() == pdi_sptr->get_max_tangential_pos_num());
}

template <typename elemT>
Viewgram<elemT>::
Viewgram(const shared_ptr<const ProjDataInfo>& pdi_sptr,
         const ViewgramIndices& ind)
  : 
  Array<2,elemT>(IndexRange2D (pdi_sptr->get_min_axial_pos_num(ind.segment_num()),
			       pdi_sptr->get_max_axial_pos_num(ind.segment_num()),
			       pdi_sptr->get_min_tangential_pos_num(),
			       pdi_sptr->get_max_tangential_pos_num())),
  proj_data_info_sptr(pdi_sptr),
  _indices(ind)
{
  assert(view_num <= proj_data_info_sptr->get_max_view_num());
  assert(view_num >= proj_data_info_sptr->get_min_view_num());
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)
}

template <typename elemT>
Viewgram<elemT>::
Viewgram(const Array<2,elemT>& p, 
	 const shared_ptr<const ProjDataInfo>& pdi_sptr,
	 const int v_num, const int s_num) 
  :
  Viewgram(p, pdi_sptr, ViewgramIndices(v_num, s_num))
{}

template <typename elemT>
Viewgram<elemT>::
Viewgram(const shared_ptr<const ProjDataInfo>& pdi_sptr,
	 const int v_num, const int s_num) 
  :
  Viewgram(pdi_sptr, ViewgramIndices(v_num, s_num))
{}

END_NAMESPACE_STIR
