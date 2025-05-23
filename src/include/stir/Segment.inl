//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, IRSL
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
Segment( const shared_ptr<const ProjDataInfo>& proj_data_info_sptr_v,const int s_num)
 :
 proj_data_info_sptr(proj_data_info_sptr_v),
 segment_num(s_num)
    {}

template <typename elemT>
int
Segment<elemT>:: get_segment_num() const
{ return segment_num; }

template <typename elemT>
shared_ptr<const ProjDataInfo>
Segment<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}

END_NAMESPACE_STIR
