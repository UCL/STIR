//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
#include "stir/error.h"

START_NAMESPACE_STIR

template <typename elemT>
SegmentBySinogram<elemT>::SegmentBySinogram(const Array<3, elemT>& v,
                                            const shared_ptr<const ProjDataInfo>& pdi_ptr,
                                            const SegmentIndices& ind)
    : Segment<elemT>(pdi_ptr, ind),
      Array<3, elemT>(v)
{
  this->check_state();
}

template <typename elemT>
SegmentBySinogram<elemT>::SegmentBySinogram(const shared_ptr<const ProjDataInfo>& pdi_ptr, const SegmentIndices& ind)
    : Segment<elemT>(pdi_ptr, ind),
      Array<3, elemT>(IndexRange3D(pdi_ptr->get_min_axial_pos_num(ind.segment_num()),
                                   pdi_ptr->get_max_axial_pos_num(ind.segment_num()),
                                   pdi_ptr->get_min_view_num(),
                                   pdi_ptr->get_max_view_num(),
                                   pdi_ptr->get_min_tangential_pos_num(),
                                   pdi_ptr->get_max_tangential_pos_num()))
{
  this->check_state();
}

template <typename elemT>
SegmentBySinogram<elemT>::SegmentBySinogram(const Array<3, elemT>& v,
                                            const shared_ptr<const ProjDataInfo>& pdi_sptr,
                                            int segment_num,
                                            int timing_pos_num)
    : SegmentBySinogram(v, pdi_sptr, SegmentIndices(segment_num, timing_pos_num))
{}

template <typename elemT>
SegmentBySinogram<elemT>::SegmentBySinogram(const shared_ptr<const ProjDataInfo>& pdi_sptr,
                                            const int segment_num,
                                            const int t_num)
    : SegmentBySinogram(pdi_sptr, SegmentIndices(segment_num, t_num))
{}

template <typename elemT>
SegmentBySinogram<elemT>::SegmentBySinogram(const SegmentByView<elemT>& s_v)

    : SegmentBySinogram<elemT>(s_v.get_proj_data_info_sptr()->create_shared_clone(), s_v.get_segment_indices())
{
  for (int r = this->get_min_axial_pos_num(); r <= this->get_max_axial_pos_num(); r++)
    set_sinogram(s_v.get_sinogram(r));
}

template <typename elemT>
void
SegmentBySinogram<elemT>::check_state() const
{
  if (!this->proj_data_info_sptr)
    error("SegmentBySinogram not properly initialised.");

  bool ok = (this->get_min_axial_pos_num() == this->get_min_index() && this->get_max_axial_pos_num() == this->get_max_index()
             && this->get_min_view_num() == (*this)[this->get_min_index()].get_min_index()
             && this->get_max_view_num() == (*this)[this->get_min_index()].get_max_index()
             && this->get_min_tangential_pos_num() == (*this)[this->get_min_index()][this->get_min_view_num()].get_min_index()
             && this->get_max_tangential_pos_num() == (*this)[this->get_min_index()][this->get_min_view_num()].get_max_index());
  if (!ok)
    error("SegmentBySinogram: inconsistent proj_data_info sizes and array sizes.");
}

template <typename elemT>
bool
SegmentBySinogram<elemT>::operator==(const Segment<elemT>& that) const
{
  return this->has_same_characteristics(that) && Array<3, elemT>::operator==(static_cast<const self_type&>(that));
}

template <typename elemT>
Viewgram<elemT>
SegmentBySinogram<elemT>::get_viewgram(int view_num) const
{
  Array<2, elemT> pre_view(IndexRange2D(this->get_min_axial_pos_num(),
                                        this->get_max_axial_pos_num(),
                                        this->get_min_tangential_pos_num(),
                                        this->get_max_tangential_pos_num()));
  for (int r = this->get_min_axial_pos_num(); r <= this->get_max_axial_pos_num(); r++)
    pre_view[r] = Array<3, elemT>::operator[](r)[view_num];
  // KT 9/12 constructed a PETSinogram before...
  //  CL&KT 15/12 added ring_difference stuff
  return Viewgram<elemT>(
      pre_view, this->proj_data_info_sptr->create_shared_clone(), view_num, this->get_segment_num(), this->get_timing_pos_num());
}

template <typename elemT>
void
SegmentBySinogram<elemT>::set_viewgram(const Viewgram<elemT>& viewgram)
{
  for (int r = this->get_min_axial_pos_num(); r <= this->get_max_axial_pos_num(); r++)
    Array<3, elemT>::operator[](r)[viewgram.get_view_num()] = viewgram[r];
}

/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void
SegmentBySinogram<elemT>::resize(const IndexRange<3>& range)
{
  if (range == this->get_index_range())
    return;

  assert(range.is_regular() == true);

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();

  // can only handle min_view==0 at the moment
  // TODO
  assert(range[ax_min].get_min_index() == 0);

  shared_ptr<ProjDataInfo> pdi_sptr = this->proj_data_info_sptr->create_shared_clone();

  pdi_sptr->set_min_axial_pos_num(ax_min, this->get_segment_num());
  pdi_sptr->set_max_axial_pos_num(ax_max, this->get_segment_num());

  pdi_sptr->set_num_views(range[ax_min].get_max_index() + 1);
  pdi_sptr->set_min_tangential_pos_num(range[ax_min][0].get_min_index());
  pdi_sptr->set_max_tangential_pos_num(range[ax_min][0].get_max_index());

  this->proj_data_info_sptr = pdi_sptr;

  Array<3, elemT>::resize(range);
}

template <typename elemT>
void
SegmentBySinogram<elemT>::grow(const IndexRange<3>& range)
{
  resize(range);
}

/*************************************
 instantiations
 *************************************/

template class SegmentBySinogram<float>;

END_NAMESPACE_STIR
