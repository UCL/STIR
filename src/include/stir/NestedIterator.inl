//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup buildblock
  \brief inline implementations for stir::NestedIterator.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

template <class topleveliterT, class GetRestRangeFunctionT>
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
NestedIterator()
{}

template <class topleveliterT, class GetRestRangeFunctionT>
void 
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
_set_rest_iters_for_current_top_level_iter()
{
  if (this->_current_top_level_iter != this->_end_top_level_iter)
    {
      this->_current_rest_iter = 
	GetRestRangeFunctionT().begin(this->_current_top_level_iter);
      this->_end_rest_iter = 
	GetRestRangeFunctionT().end(this->_current_top_level_iter);
    }
}
template <class topleveliterT, class GetRestRangeFunctionT>
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
NestedIterator(const topleveliterT& top_level_iter,
	       const topleveliterT& end_top_level_iter)
  : _current_top_level_iter(top_level_iter),
    _end_top_level_iter(end_top_level_iter)
{
  this->_set_rest_iters_for_current_top_level_iter();
}

template <class topleveliterT, class GetRestRangeFunctionT>
bool 
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
operator==(const NestedIterator<topleveliterT, GetRestRangeFunctionT>& iter2) const
{
  return
    this->_current_top_level_iter == iter2._current_top_level_iter &&
      ( this->_current_top_level_iter == this->_end_top_level_iter ||
        this->_current_rest_iter == iter2._current_rest_iter); 
    /*    
   alternative:
     comparing rest_iter is only necessary when the first iterator is not at the end. 
     This probably doesn't matter too much though as usually we are comparing with
     end_all(), in which case the top_level_iters will only be equal when we are
     at the end. So, the extra test would only occurs once in the loop over the whole
     sequence.

     A (possibly sligthly faster) implementation would be:

       _current_top_level_iter == iter2._current_top_level_iter &&
       _current_rest_iter == iter2._current_rest_iter;

     However, the above relies on the fact that incrementing the iterator
     ends up exactly in end_all(). This seems tricky to implement in general.
    */
}  
template <class topleveliterT, class GetRestRangeFunctionT>
bool 
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
operator!=(const NestedIterator<topleveliterT, GetRestRangeFunctionT>& iter2) const
{
  return !(*this == iter2);
}

template <class topleveliterT, class GetRestRangeFunctionT>
NestedIterator<topleveliterT, GetRestRangeFunctionT>& 
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
operator++()
{
  // TODO can only do assert for random-access iterators
  // assert(this->_current_top_level_iter < this->_end_top_level_iter);
  ++this->_current_rest_iter;
  if (this->_current_rest_iter == this->_end_rest_iter)
  {
    // advance the top-level iterator and reset rest_iters
    ++this->_current_top_level_iter;
    this->_set_rest_iters_for_current_top_level_iter();    
  }
  return *this;
}

template <class topleveliterT, class GetRestRangeFunctionT>
NestedIterator<topleveliterT, GetRestRangeFunctionT> 
NestedIterator<topleveliterT, GetRestRangeFunctionT>::operator++(int)
{
  const NestedIterator<topleveliterT, GetRestRangeFunctionT> was = *this;
  ++(*this);
  return was;
}


template <class topleveliterT, class GetRestRangeFunctionT>
typename NestedIterator<topleveliterT, GetRestRangeFunctionT>::reference
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
operator*() const
{
  return *this->_current_rest_iter;
}

template <class topleveliterT, class GetRestRangeFunctionT>
typename NestedIterator<topleveliterT, GetRestRangeFunctionT>::pointer
NestedIterator<topleveliterT, GetRestRangeFunctionT>::
operator->() const
{
  return &(this->operator*());
}

END_NAMESPACE_STIR
