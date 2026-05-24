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
#include "stir/error.h"
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
{
  return this->_indices.segment_num();
}

template <typename elemT>
int
Viewgram<elemT>::get_view_num() const
{
  return this->_indices.view_num();
}

template <typename elemT>
int
Viewgram<elemT>::get_timing_pos_num() const
{
  return this->_indices.timing_pos_num();
}

template <typename elemT>
int
Viewgram<elemT>::get_min_axial_pos_num() const
{
  return this->get_min_index();
}

template <typename elemT>
int
Viewgram<elemT>::get_max_axial_pos_num() const
{
  return this->get_max_index();
}

template <typename elemT>
int
Viewgram<elemT>::get_num_axial_poss() const
{
  return this->get_length();
}

template <typename elemT>
Viewgram<elemT>
Viewgram<elemT>::get_empty_copy(void) const
{
  Viewgram<elemT> copy(proj_data_info_sptr, get_viewgram_indices());
  return copy;
}

template <typename elemT>
Viewgram<elemT>::Viewgram(const Array<2, elemT>& p, const shared_ptr<const ProjDataInfo>& pdi_sptr, const ViewgramIndices& ind)
    : Array<2, elemT>(p),
      DataWithProjDataInfo(pdi_sptr),
      _indices(ind)
{
  if (!pdi_sptr)
    error("Sinogram constructed with empty proj_data_info");
  if (ind.segment_num() > this->get_max_segment_num() || ind.segment_num() < this->get_min_segment_num()
      || ind.view_num() > this->get_max_view_num() || ind.view_num() < this->get_min_view_num()
      || ind.timing_pos_num() > this->get_max_tof_pos_num() || ind.timing_pos_num() < this->get_min_tof_pos_num())
    error("Viewgram constructed with out-of-range indices");
  const bool ok = (p.get_min_index() == this->get_min_axial_pos_num() && p.get_max_index() == this->get_max_axial_pos_num()
                   && (p.size() == 0
                       || (p[p.get_min_index()].get_min_index() == this->get_min_tangential_pos_num()
                           && p[p.get_min_index()].get_max_index() == this->get_max_tangential_pos_num())));
  if (!ok)
    error("Viewgram constructed with array with dimensions that are inconsistent with the proj_data_info");
}

template <typename elemT>
Viewgram<elemT>::Viewgram(const shared_ptr<const ProjDataInfo>& pdi_sptr, const ViewgramIndices& ind)
    : Viewgram(Array<2, elemT>(IndexRange2D(pdi_sptr->get_min_axial_pos_num(ind.segment_num()),
                                            pdi_sptr->get_max_axial_pos_num(ind.segment_num()),
                                            pdi_sptr->get_min_tangential_pos_num(),
                                            pdi_sptr->get_max_tangential_pos_num())),
               pdi_sptr,
               ind)
{}

template <typename elemT>
Viewgram<elemT>::Viewgram(
    const Array<2, elemT>& p, const shared_ptr<const ProjDataInfo>& pdi_sptr, const int v_num, const int s_num, const int t_num)
    : Viewgram(p, pdi_sptr, ViewgramIndices(v_num, s_num, t_num))
{}

template <typename elemT>
Viewgram<elemT>::Viewgram(const shared_ptr<const ProjDataInfo>& pdi_sptr, const int v_num, const int s_num, const int t_num)
    : Viewgram(pdi_sptr, ViewgramIndices(v_num, s_num, t_num))
{}

END_NAMESPACE_STIR
