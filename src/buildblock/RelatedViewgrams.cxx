//
// $Id$: $Date$
//
/*!

  \file

  \brief Implementations for class RelatedViewgrams

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "RelatedViewgrams.h"

#ifdef _MSC_VER
// disable warning that constructor with PMessage is not implemented
#pragma warning(disable: 4661)
#endif // _MSC_VER

START_NAMESPACE_TOMO


// a function which is called internally to see if the object is valid
template <typename elemT>
void RelatedViewgrams<elemT>::debug_check_state() const
{
  // KT 09/03/99 can't use any methods of RelatedViewgrams here, as
  // this causes an infinite recursion with check_state
  if (viewgrams.size() == 0)
    return;

  vector<ViewSegmentNumbers> pairs;
  symmetries_used->get_related_view_segment_numbers(
    pairs, 
    ViewSegmentNumbers(
       viewgrams[0].get_view_num(),
       viewgrams[0].get_segment_num()
	) );

  assert(pairs.size() == viewgrams.size());
  for (unsigned int i=0; i<viewgrams.size(); i++)
  {
    assert(viewgrams[i].get_view_num() == pairs[i].view_num());
    assert(viewgrams[i].get_segment_num() == pairs[i].segment_num());
  }

  for (unsigned int i=1; i<viewgrams.size(); i++)
  {
    assert(*(viewgrams[i].get_proj_data_info_ptr()) ==
           *(viewgrams[0].get_proj_data_info_ptr()));
  }

}


template <typename elemT>
RelatedViewgrams<elemT> RelatedViewgrams<elemT>::get_empty_copy() const
{
  check_state();

  vector<Viewgram<elemT> > empty_viewgrams;
  empty_viewgrams.reserve(viewgrams.size());
  // TODO optimise to get shared proj_data_info_ptr
  for (unsigned int i=0; i<viewgrams.size(); i++)
    empty_viewgrams.push_back(viewgrams[i].get_empty_copy());

  return RelatedViewgrams<elemT>(empty_viewgrams,
                          symmetries_used);
}

/* 
  TODO
#include "zoom.h"

template <typename elemT>
void RelatedViewgrams<elemT>::zoom(const float zoom, const float Xoffp, const float Yoffp,
            const int size, const float itophi)
{
  check_state();

  for (vector<Viewgram>::iterator iter= viewgrams.begin();
       iter != viewgrams.end();
       iter++)
    zoom_viewgram((*iter),  zoom, Xoffp, Yoffp, size, itophi);

  check_state();
}
*/
/*
template <typename elemT>
void RelatedViewgrams<elemT>::grow_num_bins(const int new_min_bin_num, 
				     const int new_max_bin_num)
{
  for (vector<Viewgram>::iterator iter= viewgrams.begin();
       iter != viewgrams.end();
       iter++)
	 (*iter).grow_width(new_min_bin_num, new_max_bin_num);
}
*/

/*************************************
 instantiations
 *************************************/

template RelatedViewgrams<float>;

END_NAMESPACE_TOMO
