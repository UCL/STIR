//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Implementations for non-inline functions of class stir::Viewgram

  \author Kris Thielemans
  \author PARAPET project

*/

#include "stir/Viewgram.h"
#include "stir/IndexRange2D.h"
#include "stir/error.h"
#include "stir/format.h"

#ifdef _MSC_VER
// disable warning that not all functions have been implemented when instantiating
#  pragma warning(disable : 4661)
#endif // _MSC_VER

using std::string;

START_NAMESPACE_STIR

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

template <typename elemT>
bool
Viewgram<elemT>::has_same_characteristics(self_type const& other, string& explanation) const
{
  if (*this->get_proj_data_info_sptr() != *other.get_proj_data_info_sptr())
    {
      explanation = format("Differing projection data info:\n{}\n-------- vs-------\n {}",
                           this->get_proj_data_info_sptr()->parameter_info(),
                           other.get_proj_data_info_sptr()->parameter_info());
      return false;
    }
  if (this->get_view_num() != other.get_view_num())
    {
      explanation = format("Differing view number: {} vs {}", this->get_view_num(), other.get_view_num());
      return false;
    }
  if (this->get_segment_num() != other.get_segment_num())
    {
      explanation = format("Differing segment number: {} vs {}", this->get_segment_num(), other.get_segment_num());
      return false;
    }
  if (this->get_timing_pos_num() != other.get_timing_pos_num())
    {
      explanation = format("Differing timing position index: {} vs {}", this->get_timing_pos_num(), other.get_timing_pos_num());
      return false;
    }
  return true;
}

template <typename elemT>
bool
Viewgram<elemT>::has_same_characteristics(self_type const& other) const
{
  std::string explanation;
  return this->has_same_characteristics(other, explanation);
}

template <typename elemT>
bool
Viewgram<elemT>::operator==(const self_type& that) const
{
  return this->has_same_characteristics(that) && base_type::operator==(that);
}

template <typename elemT>
bool
Viewgram<elemT>::operator!=(const self_type& that) const
{
  return !((*this) == that);
}

/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void
Viewgram<elemT>::resize(const IndexRange<2>& range)
{
  if (range == this->get_index_range())
    return;

  assert(range.is_regular() == true);

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();

  shared_ptr<ProjDataInfo> pdi_sptr(proj_data_info_sptr->clone());

  pdi_sptr->set_min_axial_pos_num(ax_min, get_segment_num());
  pdi_sptr->set_max_axial_pos_num(ax_max, get_segment_num());
  pdi_sptr->set_min_tangential_pos_num(range[ax_min].get_min_index());
  pdi_sptr->set_max_tangential_pos_num(range[ax_min].get_max_index());

  proj_data_info_sptr = pdi_sptr;

  Array<2, elemT>::resize(range);
}

/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void
Viewgram<elemT>::grow(const IndexRange<2>& range)
{
  resize(range);
}

/******************************
 instantiations
 ****************************/

template class Viewgram<float>;

END_NAMESPACE_STIR
