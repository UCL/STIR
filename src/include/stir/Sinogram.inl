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
#include "stir/error.h"
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
Sinogram<elemT>
Sinogram<elemT>::get_empty_copy(void) const
{
  Sinogram<elemT> copy(this->get_proj_data_info_sptr(), get_sinogram_indices());
  return copy;
}

template <typename elemT>
Sinogram<elemT>::Sinogram(const Array<2, elemT>& p, const shared_ptr<const ProjDataInfo>& pdi_sptr, const SinogramIndices& ind)
    : Array<2, elemT>(p),
      DataWithProjDataInfo(pdi_sptr),
      _indices(ind)
{
  if (!pdi_sptr)
    error("Sinogram constructed with empty proj_data_info");
  if (ind.segment_num() > this->get_max_segment_num() || ind.segment_num() < this->get_min_segment_num()
      || ind.axial_pos_num() > this->get_max_axial_pos_num(ind.segment_num())
      || ind.axial_pos_num() < this->get_min_axial_pos_num(ind.segment_num())
      || ind.timing_pos_num() > this->get_max_tof_pos_num() || ind.timing_pos_num() < this->get_min_tof_pos_num())
    error("Sinogram constructed with out-of-range indices");
  const bool ok = (p.get_min_index() == this->get_min_view_num() && p.get_max_index() == this->get_max_view_num()
                   && (p.size() == 0
                       || (p[p.get_min_index()].get_min_index() == this->get_min_tangential_pos_num()
                           && p[p.get_min_index()].get_max_index() == this->get_max_tangential_pos_num())));
  if (!ok)
    error("Sinogram constructed with array with dimensions that are inconsistent with the proj_data_info");
}

template <typename elemT>
Sinogram<elemT>::Sinogram(const shared_ptr<const ProjDataInfo>& pdi_ptr, const SinogramIndices& ind)
    : Sinogram(Array<2, elemT>(IndexRange2D(pdi_ptr->get_min_view_num(),
                                            pdi_ptr->get_max_view_num(),
                                            pdi_ptr->get_min_tangential_pos_num(),
                                            pdi_ptr->get_max_tangential_pos_num())),
               pdi_ptr,
               ind)
{}

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
