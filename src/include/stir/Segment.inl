//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, IRSL
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief implementation of inline functions of class stir::Segment

  \author Kris Thielemans
  \author PARAPET project


*/
#include "stir/Sinogram.h"

START_NAMESPACE_STIR

template <typename elemT>
Segment<elemT>::
Segment( const shared_ptr<const ProjDataInfo>& proj_data_info_sptr_v,const SegmentIndices& ind)
 :
 proj_data_info_sptr(proj_data_info_sptr_v),
 _indices(ind)
    {}

template <typename elemT>
SegmentIndices
Segment<elemT>:: get_segment_indices() const
{ return _indices; }

template <typename elemT>
int
Segment<elemT>:: get_segment_num() const
{ return _indices.segment_num(); }

template <typename elemT>
shared_ptr<const ProjDataInfo>
Segment<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}

END_NAMESPACE_STIR
