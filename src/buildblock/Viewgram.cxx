//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for non-inline functions of class Viewgram

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "Viewgram.h"

#ifdef _MSC_VER
// disable warning that not all functions have been implemented when instantiating
#pragma warning(disable: 4661)
#endif // _MSC_VER
START_NAMESPACE_TOMO


/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
Viewgram<elemT>::
grow(const IndexRange<2>& range)
{   
  if (range == get_index_range())
    return;

  const int ax_min = get_min_axial_pos_num();
  const int ax_max = get_max_axial_pos_num();

  // can not set axial_pos_num of ProjDataInfo at the moment
  // TODO
  assert(range.get_min_index() == ax_min);
  assert(range.get_max_index() == ax_max);
  
  ProjDataInfo* pdi_ptr = proj_data_info_ptr->clone();
  
  pdi_ptr->set_min_tangential_pos_num(range[ax_min].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[ax_min].get_max_index());

  proj_data_info_ptr = pdi_ptr;

  Array<2,elemT>::grow(range);
	
}


/******************************
 instantiations
 ****************************/

template Viewgram<float>;

END_NAMESPACE_TOMO
