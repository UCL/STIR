//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Implementations for non-inline functions of class Viewgram

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/Viewgram.h"

#ifdef _MSC_VER
// disable warning that not all functions have been implemented when instantiating
#pragma warning(disable: 4661)
#endif // _MSC_VER
START_NAMESPACE_STIR


/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
Viewgram<elemT>::
grow(const IndexRange<2>& range)
{   
  if (range == this->get_index_range())
    return;

  assert(range.is_regular()==true);

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();
  
  ProjDataInfo* pdi_ptr = proj_data_info_ptr->clone();

  pdi_ptr->set_min_axial_pos_num(ax_min, get_segment_num());
  pdi_ptr->set_max_axial_pos_num(ax_max, get_segment_num());
  pdi_ptr->set_min_tangential_pos_num(range[ax_min].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[ax_min].get_max_index());

  proj_data_info_ptr = pdi_ptr;

  Array<2,elemT>::grow(range);
	
}


/******************************
 instantiations
 ****************************/

template class Viewgram<float>;

END_NAMESPACE_STIR
