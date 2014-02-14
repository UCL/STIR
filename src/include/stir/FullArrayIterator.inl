//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
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
  \ingroup Array
  \brief inline implementations for stir::FullArrayIterator.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project



*/

START_NAMESPACE_STIR

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
  FullArrayIterator()
  : current_top_level_iter(0),
    last_top_level_iter(0),
    current_rest_iter(),
    last_rest_iter()
{}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
  FullArrayIterator(unsigned 
#ifndef NDEBUG
                    param // only give the parameter a name in debug mode to avoid compiler warning
#endif
                    )
  : current_top_level_iter(0),
    last_top_level_iter(0),
    current_rest_iter(0),
    last_rest_iter(0)
{
  assert(param==0);
}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
  FullArrayIterator(const topleveliterT& top_level_iter,
                    const topleveliterT& last_top_level_iter,
                    const restiterT& rest_iter,
                    const restiterT& last_rest_iter)
  : current_top_level_iter(top_level_iter),
    last_top_level_iter(last_top_level_iter),
    current_rest_iter(rest_iter),  
    last_rest_iter(last_rest_iter)
{}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr> 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
FullArrayIterator(const FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& iter2)
  : current_top_level_iter(iter2.current_top_level_iter),
    last_top_level_iter(iter2.last_top_level_iter),
    current_rest_iter(iter2.current_rest_iter),
    last_rest_iter(iter2.last_rest_iter)
{}


template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
bool 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>
  ::operator==(const FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& iter2) const
{
  return
    current_top_level_iter == iter2.current_top_level_iter &&
    current_rest_iter == iter2.current_rest_iter;

  /* alternative:
     comparing rest_iter is only necessary when the first iterator is not 
     at the end. However, this is the most common case, so the 
     implementation above is faster as it has one test less.
     However, the above relies on the fact that incrementing the iterator
     ends up exactly in end_all().

    current_top_level_iter == iter2.current_top_level_iter &&
      ( current_top_level_iter != last_top_level_iterator &&
        current_rest_iter == iter2.current_rest_iter); 
    */
}  
template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
bool 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>
  ::operator!=(const FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& iter2) const
{
  return !(*this == iter2);
}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
typename FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::reference
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator*() const
{
  return *current_rest_iter;
}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
typename FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::pointer
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator->() const
{
  return &(operator*());
}

/*! We make sure that incrementing the full_iterator ends up in
    (last_top_level_iter, last_top_level_iter, 0, 0).
    This <b>has to</b> represent \c end_all() of this full_iterator.
*/
template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator++()
{
  assert(current_top_level_iter < last_top_level_iter);
  ++current_rest_iter;
  if (current_rest_iter == last_rest_iter)
  {
    ++current_top_level_iter;        
    if (current_top_level_iter != (last_top_level_iter))
    {
      current_rest_iter = (*current_top_level_iter).begin_all();
      last_rest_iter = (*current_top_level_iter).end_all();
    }
    else
    {
      current_rest_iter = restiterT(0);
      last_rest_iter = restiterT(0);
    }
  }
  return *this;
}

template <class topleveliterT, class restiterT, class elemT, class _Ref, class _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr> 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator++(int)
{
  FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr> was = *this;
  ++(*this);
  return was;
}


END_NAMESPACE_STIR
