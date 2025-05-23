//
//
/*!

  \file
  \ingroup projdata
  \brief inline implementations for class stir::RelatedViewgrams

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/ViewSegmentNumbers.h"

START_NAMESPACE_STIR

template <typename elemT>
RelatedViewgrams<elemT>::RelatedViewgrams() :
     viewgrams(), symmetries_used()
     {}
  
template <typename elemT>
RelatedViewgrams<elemT>::RelatedViewgrams(const std::vector<Viewgram<elemT> >& viewgrams,
                   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used)
		   : viewgrams(viewgrams), 
		     symmetries_used(symmetries_used)
  {
    check_state();
  }

template <typename elemT>
void RelatedViewgrams<elemT>::check_state() const
{
#ifndef NDEBUG
  debug_check_state();
#endif
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_num_viewgrams() const
{
  check_state();
  return static_cast<int>(viewgrams.size());
}


template <typename elemT>
int RelatedViewgrams<elemT>::get_basic_view_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_view_num();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_basic_segment_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_segment_num();
}

template <typename elemT>
ViewSegmentNumbers RelatedViewgrams<elemT>::
get_basic_view_segment_num() const
{
  return ViewSegmentNumbers(get_basic_view_num(), get_basic_segment_num());
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_num_axial_poss() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_num_axial_poss();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_num_tangential_poss() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_num_tangential_poss();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_min_axial_pos_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_min_axial_pos_num();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_max_axial_pos_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_max_axial_pos_num();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_min_tangential_pos_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_min_tangential_pos_num();
}

template <typename elemT>
int RelatedViewgrams<elemT>::get_max_tangential_pos_num() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_max_tangential_pos_num();
}

template <typename elemT>
shared_ptr<const ProjDataInfo>
RelatedViewgrams<elemT>::
get_proj_data_info_sptr() const
{
  assert(viewgrams.size()>0);
  check_state();
  return viewgrams[0].get_proj_data_info_sptr();
}

template <typename elemT>
const DataSymmetriesForViewSegmentNumbers * 
RelatedViewgrams<elemT>::get_symmetries_ptr() const
{
  return symmetries_used.get();
}

template <typename elemT>
shared_ptr<DataSymmetriesForViewSegmentNumbers>
RelatedViewgrams<elemT>::get_symmetries_sptr() const
{
  return symmetries_used;
}

template <typename elemT>
typename RelatedViewgrams<elemT>::iterator 
RelatedViewgrams<elemT>::begin()
{ return viewgrams.begin();}

template <typename elemT>
typename RelatedViewgrams<elemT>::iterator 
RelatedViewgrams<elemT>::end()
{return viewgrams.end();}

template <typename elemT>
typename RelatedViewgrams<elemT>::const_iterator 
RelatedViewgrams<elemT>::begin() const
{return viewgrams.begin();}

template <typename elemT>
typename RelatedViewgrams<elemT>::const_iterator 
RelatedViewgrams<elemT>::end() const
{return viewgrams.end();}


END_NAMESPACE_STIR
