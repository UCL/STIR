//
// $Id$
//
/*!

  \file
  \ingroup projdata
  \brief Implementations for non-inline functions of class Sinogram

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
Sinogram<elemT>::
grow(const IndexRange<2>& range)
{   
  if (range == this->get_index_range())
    return;

  // can only handle min_view==0 at the moment
  // TODO
  assert(range.get_min_index() == 0);
  
  ProjDataInfo* pdi_ptr = proj_data_info_ptr->clone();
  
  pdi_ptr->set_num_views(range.get_max_index() + 1);
  pdi_ptr->set_min_tangential_pos_num(range[0].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[0].get_max_index());

  proj_data_info_ptr = pdi_ptr;

  Array<2,elemT>::grow(range);
	
}


/******************************
 instantiations
 ****************************/

template class Sinogram<float>;

END_NAMESPACE_STIR
