//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata
  \brief Implementations for non-inline functions of class stir::Segment

  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/Segment.h"
#include "stir/SegmentByView.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/format.h"

using std::string;
START_NAMESPACE_STIR

template <typename elemT>
bool
Segment<elemT>::has_same_characteristics(self_type const& other, string& explanation) const
{
  if (typeid(*this) != typeid(other))
    {
      explanation = format("Differing data types:{} vs {}", typeid(*this).name(), typeid(other).name());
      return false;
    }
  if (*this->get_proj_data_info_sptr() != *other.get_proj_data_info_sptr())
    {
      explanation = format("Differing projection data info:\n{}\n-------- vs-------\n {}",
                           this->get_proj_data_info_sptr()->parameter_info(),
                           other.get_proj_data_info_sptr()->parameter_info());
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
Segment<elemT>::has_same_characteristics(self_type const& other) const
{
  std::string explanation;
  return this->has_same_characteristics(other, explanation);
}

template <typename elemT>
bool
Segment<elemT>::operator!=(const self_type& that) const
{
  return !((*this) == that);
}

/*************************************
 instantiations
 *************************************/

template class Segment<float>;

END_NAMESPACE_STIR
