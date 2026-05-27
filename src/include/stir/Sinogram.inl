//
//
/*
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Implementations of inline functions of class stir::Sinogram

  \author Kris Thielemans


*/

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

END_NAMESPACE_STIR
