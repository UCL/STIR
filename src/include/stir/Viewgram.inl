//
//
/*!

  \file
  \ingroup projdata

  \brief Inline implementations of class stir::Viewgram

  \author Kris Thielemans
*/
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

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

END_NAMESPACE_STIR
