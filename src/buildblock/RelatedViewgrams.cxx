//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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

  \brief Implementations for class stir::RelatedViewgrams

  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/RelatedViewgrams.h"
#include "boost/format.hpp"

#ifdef _MSC_VER
// disable warning that constructor with PMessage is not implemented
#pragma warning(disable: 4661)
#endif // _MSC_VER

START_NAMESPACE_STIR


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

template<typename elemT>
bool
RelatedViewgrams<elemT>::
has_same_characteristics(self_type const& other,
			 string& explanation) const
{
  using boost::format;
  using boost::str;

  if (*this->get_proj_data_info_ptr() !=
      *other.get_proj_data_info_ptr())
    {
      explanation = 
	str(format("Differing projection data info:\n%1%\n-------- vs-------\n %2%")
	    % this->get_proj_data_info_ptr()->parameter_info()
	    % other.get_proj_data_info_ptr()->parameter_info()
	    );
      return false;
    }
  if (*this->get_symmetries_ptr() !=
      *other.get_symmetries_ptr())
    {
      explanation = 
	str(format("Differing symmetries")
	    );
      return false;
    }
  if (this->get_basic_view_num() !=
      other.get_basic_view_num())
    {
      explanation = 
	str(format("Differing basic view number: %1% vs %2%")
	    % this->get_basic_view_num()
	    % other.get_basic_view_num()
	    );
      return false;
    }
  if (this->get_basic_segment_num() !=
      other.get_basic_segment_num())
    {
      explanation = 
	str(format("Differing basic segment number: %1% vs %2%")
	    % this->get_basic_segment_num()
	    % other.get_basic_segment_num()
	    );
      return false;
    }
  return true;
}

template<typename elemT>
bool
RelatedViewgrams<elemT>::
has_same_characteristics(self_type const& other) const
{
  std::string explanation;
  return this->has_same_characteristics(other, explanation);
}

template<typename elemT>
bool 
RelatedViewgrams<elemT>::
operator ==(const self_type& that) const
{
  return
    this->has_same_characteristics(that) &&
    std::equal(this->begin(), this->end(), that.begin());
}
  
template<typename elemT>
bool 
RelatedViewgrams<elemT>::
operator !=(const self_type& that) const
{
  return !((*this) == that);
}

/*! \warning: this uses multiplication according to elemT (careful for overflow for integer types!) */
template <typename elemT>
RelatedViewgrams<elemT>& 
RelatedViewgrams<elemT>::
operator*= (const elemT f)
{
  for (iterator iter = begin(); iter != end(); ++iter)
    *iter *= f;
  return *this;
}

/*! \warning: this uses division according to elemT (i.e. no rounding or so) */
template <typename elemT>   
RelatedViewgrams<elemT>&
RelatedViewgrams<elemT>::
operator/= (const elemT f)
{
  assert(f!=0);

  for (iterator iter = begin(); iter != end(); ++iter)
    *iter /= f;
  return *this;
}

/*! \warning: this uses addition according to elemT (careful with overflow with integer types!) */
template <typename elemT>
RelatedViewgrams<elemT>& 
RelatedViewgrams<elemT>::
operator+= (const elemT f)
{
  for (iterator iter = begin(); iter != end(); ++iter)
    *iter += f;
  return *this;
}

/*! \warning: this uses subtraction according to elemT (careful with unsigned types!) */
template <typename elemT>
RelatedViewgrams<elemT>& 
RelatedViewgrams<elemT>::
operator-= (const elemT f)
{
  for (iterator iter = begin(); iter != end(); ++iter)
    *iter -= f;
  return *this;
}


/*! \warning: this uses multiplication according to elemT (careful for overflow for integer types!) */
template <typename elemT>   
RelatedViewgrams<elemT>&
RelatedViewgrams<elemT>::
operator*= (const RelatedViewgrams<elemT>& arg)
{
  assert(get_num_viewgrams() == arg.get_num_viewgrams());
  iterator iter = begin(); 
  const_iterator arg_iter = arg.begin();
  for ( ; iter != end(); ++iter, ++arg_iter)
    *iter *= *arg_iter;
  return *this;
}

/*! \warning: this uses division according to elemT (i.e. no rounding or so) */
template <typename elemT>   
RelatedViewgrams<elemT>&
RelatedViewgrams<elemT>::
operator/= (const RelatedViewgrams<elemT>& arg)
{
  assert(get_num_viewgrams() == arg.get_num_viewgrams());
  iterator iter = begin(); 
  const_iterator arg_iter = arg.begin();
  for ( ; iter != end(); ++iter, ++arg_iter)
    *iter /= *arg_iter;
  return *this;
}


/*! \warning: this uses addition according to elemT (careful with overflow with integer types!) */
template <typename elemT>   
RelatedViewgrams<elemT>&
RelatedViewgrams<elemT>::
operator+= (const RelatedViewgrams<elemT>& arg)
{
  assert(get_num_viewgrams() == arg.get_num_viewgrams());
  iterator iter = begin(); 
  const_iterator arg_iter = arg.begin();
  for ( ; iter != end(); ++iter, ++arg_iter)
    *iter += *arg_iter;
  return *this;
}

/*! \warning: this uses subtraction according to elemT (careful with unsigned types!) */
template <typename elemT>   
RelatedViewgrams<elemT>&
RelatedViewgrams<elemT>::
operator-= (const RelatedViewgrams<elemT>& arg)
{
  assert(get_num_viewgrams() == arg.get_num_viewgrams());
  iterator iter = begin(); 
  const_iterator arg_iter = arg.begin();
  for ( ; iter != end(); ++iter, ++arg_iter)
    *iter -= *arg_iter;
  return *this;
}



template <typename elemT>
elemT 
RelatedViewgrams<elemT>::
find_max() const
{
  Array<1,elemT> max_per_viewgram(get_num_viewgrams());
  typename Array<1,elemT>::iterator max_iter = max_per_viewgram.begin();
  const_iterator iter = begin(); 
  while (iter != end())
  {
    *max_iter = iter->find_max();
    ++iter; ++ max_iter;
  }
  return max_per_viewgram.find_max();
}

template <typename elemT>
elemT 
RelatedViewgrams<elemT>::
find_min() const
{
  Array<1,elemT> min_per_viewgram(get_num_viewgrams());
  typename Array<1,elemT>::iterator min_iter = min_per_viewgram.begin();
  const_iterator iter = begin(); 
  while (iter != end())
  {
    *min_iter = iter->find_min();
    ++iter; ++ min_iter;
  }
  return min_per_viewgram.find_min();
}


template <typename elemT>
void 
RelatedViewgrams<elemT>::fill(const elemT &n)
{
 for (iterator iter = begin(); iter != end(); ++iter)
    iter->fill(n);
}

/*! 
   This function is necessary because it modifies the size of
   each viewgram sequentially. This is not allowed by an external 
   function, and leads to different proj_data_info_ptrs anyway.
   So, it would be caught by an assert at some point.
   */
template <typename elemT>
void RelatedViewgrams<elemT>::
grow(const IndexRange<2>& range)
{
  check_state();

  if (begin()==end())
    return;

  if (range == begin()->get_index_range())
    return;

  assert(range.is_regular()==true);

  // first construct a new appropriate ProjDataInfo object

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();
  
  shared_ptr<ProjDataInfo> pdi_ptr(get_proj_data_info_ptr()->clone());

  // set axial_pos range for all segments
  for (const_iterator iter= begin();
       iter != end();
       ++iter)
  {
    pdi_ptr->set_min_axial_pos_num(ax_min, iter->get_segment_num());
    pdi_ptr->set_max_axial_pos_num(ax_max, iter->get_segment_num());
  }
  pdi_ptr->set_min_tangential_pos_num(range[ax_min].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[ax_min].get_max_index());
  shared_ptr<ProjDataInfo> pdi_shared_ptr = pdi_ptr;

  // now resize each viewgram 
  // this will not set their respective proj_data_info_ptr correctly though,
  // so, we have to construct new viewgrams for this
  for (iterator iter= begin();
       iter != end();
       ++iter)
  {
    iter->grow(range);
    *iter = Viewgram<elemT>(*iter, pdi_shared_ptr, 
                            iter->get_view_num(), iter->get_segment_num());
  }

  check_state();
}
/* 
  TODO
#include "stir/zoom.h"

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

template class RelatedViewgrams<float>;

END_NAMESPACE_STIR
