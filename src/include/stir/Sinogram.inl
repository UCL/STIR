//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007,Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
SinogramIndices
Sinogram<elemT>::get_sinogram_indices() const
{
  return this->_indices;
}

template <typename elemT>
int
Sinogram<elemT>::get_segment_num() const
{
  return this->_indices.segment_num();
}

template <typename elemT>
int
Sinogram<elemT>::get_axial_pos_num() const
{
  return this->_indices.axial_pos_num();
}

template <typename elemT>
int
Sinogram<elemT>::get_timing_pos_num() const
{
  return this->_indices.timing_pos_num();
}

template <typename elemT>
int
Sinogram<elemT>::get_min_view_num() const
{
  return this->get_min_index();
}

template <typename elemT>
int
Sinogram<elemT>::get_max_view_num() const
{
  return this->get_max_index();
}

template <typename elemT>
int
Sinogram<elemT>::get_num_views() const
{
  return this->get_length();
}

template <typename elemT>
int
Sinogram<elemT>::get_num_tangential_poss() const
{
  return this->get_length() == 0 ? 0 : (*this)[get_min_view_num()].get_length();
}

template <typename elemT>
int
Sinogram<elemT>::get_min_tangential_pos_num() const
{
  return this->get_length() == 0 ? 0 : (*this)[get_min_view_num()].get_min_index();
}

template <typename elemT>
int
Sinogram<elemT>::get_max_tangential_pos_num() const
{
  return this->get_length() == 0 ? 0 : (*this)[get_min_view_num()].get_max_index();
}

template <typename elemT>
Sinogram<elemT>
Sinogram<elemT>::get_empty_copy(void) const
{
  Sinogram<elemT> copy(proj_data_info_ptr, get_sinogram_indices());
  return copy;
}

template <typename elemT>
shared_ptr<const ProjDataInfo>
Sinogram<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_ptr;
}

template <typename elemT>
Sinogram<elemT>::Sinogram(const Array<2, elemT>& p, const shared_ptr<const ProjDataInfo>& pdi_ptr, const SinogramIndices& ind)
    : Array<2, elemT>(p),
      proj_data_info_ptr(pdi_ptr),
      _indices(ind)
{
  assert(ind.axial_pos_num() <= proj_data_info_ptr->get_max_axial_pos_num(ind.segment_num()));
  assert(ind.axial_pos_num() >= proj_data_info_ptr->get_min_axial_pos_num(ind.segment_num()));
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)

  assert(get_min_view_num() == pdi_ptr->get_min_view_num());
  assert(get_max_view_num() == pdi_ptr->get_max_view_num());
  assert(get_min_tangential_pos_num() == pdi_ptr->get_min_tangential_pos_num());
  assert(get_max_tangential_pos_num() == pdi_ptr->get_max_tangential_pos_num());
}

template <typename elemT>
Sinogram<elemT>::Sinogram(const shared_ptr<const ProjDataInfo>& pdi_ptr, const SinogramIndices& ind)
    : Array<2, elemT>(IndexRange2D(pdi_ptr->get_min_view_num(),
                                   pdi_ptr->get_max_view_num(),
                                   pdi_ptr->get_min_tangential_pos_num(),
                                   pdi_ptr->get_max_tangential_pos_num())),
      proj_data_info_ptr(pdi_ptr),
      _indices(ind)
{
  assert(ind.axial_pos_num() <= proj_data_info_ptr->get_max_axial_pos_num(ind.segment_num()));
  assert(ind.axial_pos_num() >= proj_data_info_ptr->get_min_axial_pos_num(ind.segment_num()));
  // segment_num is already checked by doing get_max_axial_pos_num(s_num)
}

template <typename elemT>
Sinogram<elemT>::Sinogram(const Array<2, elemT>& p,
                          const shared_ptr<const ProjDataInfo>& pdi_sptr,
                          const int ax_pos_num,
                          const int s_num,
                          const int t_num)
    : Sinogram(p, pdi_sptr, SinogramIndices(ax_pos_num, s_num, t_num))
{}

template <typename elemT>
Sinogram<elemT>::Sinogram(const shared_ptr<const ProjDataInfo>& pdi_sptr, const int ax_pos_num, const int s_num, const int t_num)
    : Sinogram(pdi_sptr, SinogramIndices(ax_pos_num, s_num, t_num))
{}

END_NAMESPACE_STIR
