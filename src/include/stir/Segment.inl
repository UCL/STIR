//
// $Id$
//
/*!
  \file
  \ingroup projdata
  \brief implementation of inline functions of class Segment

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/Sinogram.h"

START_NAMESPACE_STIR

template <typename elemT>
Segment<elemT>::
Segment( const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,const int s_num)
 :
 proj_data_info_ptr(proj_data_info_ptr_v),
 segment_num(s_num)
    {}

template <typename elemT>
int
Segment<elemT>:: get_segment_num() const
{ return segment_num; }


template <typename elemT>
const ProjDataInfo*
Segment<elemT>::get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}

template <typename elemT>
void 
Segment<elemT>::set_sinogram(const Sinogram<elemT>& s)
{ set_sinogram(s,s.get_num_axial_poss()); }
 
END_NAMESPACE_STIR
