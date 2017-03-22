//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, IRSL
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

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
Segment( const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,const int s_num, const int t_num)
 :
 proj_data_info_ptr(proj_data_info_ptr_v),
 segment_num(s_num),
 timing_pos_num(t_num)
    {}

template <typename elemT>
int
Segment<elemT>:: get_segment_num() const
{ return segment_num; }

template <typename elemT>
int
Segment<elemT>:: get_timing_pos_num() const
{ return timing_pos_num; }

template <typename elemT>
const ProjDataInfo*
Segment<elemT>::get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}

template <typename elemT>
shared_ptr<ProjDataInfo>
Segment<elemT>::get_proj_data_info_sptr() const
{
  return proj_data_info_ptr;
}

END_NAMESPACE_STIR
