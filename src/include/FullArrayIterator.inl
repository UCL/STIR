//
// $Id$: $Date$
//

/*!
  \file 
 
  \brief inline implementations for FullArrayIterator.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

START_NAMESPACE_TOMO

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
  FullArrayIterator()
  : current_top_level_iter(0),
    last_top_level_iter(0),
    current_rest_iter(),
    last_rest_iter()
{}

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
  FullArrayIterator(const topleveliterT& top_level_iter,
                    const topleveliterT& last_top_level_iter,
                  const restiterT& rest_iter)
  : current_top_level_iter(top_level_iter),
    last_top_level_iter(last_top_level_iter),
    current_rest_iter(rest_iter)  
{
  if (top_level_iter == last_top_level_iter)
    last_rest_iter = restiterT();
  else
    last_rest_iter = (*top_level_iter).end_all();
}


template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr> 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::
FullArrayIterator(const FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& iter2)
  : current_top_level_iter(iter2.current_top_level_iter),
    last_top_level_iter(iter2.last_top_level_iter),
    current_rest_iter(iter2.current_rest_iter),
    last_rest_iter(iter2.last_rest_iter)
{}

/*! We make sure that incrementing the full_iterator ends up in
    (last_top_level_iter - 1, *(last_top_level_iter - 1).end_all()).
    This SHOULD represent end_all() of this full_iterator.
*/
template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator++()
{
  assert(current_top_level_iter < last_top_level_iter);
  ++current_rest_iter;

  if (current_rest_iter == last_rest_iter
      && current_top_level_iter != (last_top_level_iter - 1))
  {
    ++current_top_level_iter;
    current_rest_iter = (*current_top_level_iter).begin_all();
    last_rest_iter = (*current_top_level_iter).end_all();
  }
  return *this;
}

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr> 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator++(int)
{
  FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr> was = *this;
  ++(*this);
  return was;
}

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
bool 
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>
  ::operator==(const FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>& iter2) const
{
  return
    current_top_level_iter == iter2.current_top_level_iter &&
    current_rest_iter == iter2.current_rest_iter;

  /* alternative:
     comparing rest_iter is only necessary when the first iterator is not 
     at the end. However, this is the most common case, so the actual
     implementation is faster as it has one test less.
     However, the above relies on the fact that incrementing the iterator
     ends up exactly in end_all().

    current_top_level_iter == iter2.current_top_level_iter &&
      ( current_top_level_iter != last_top_level_iterator &&
        current_rest_iter == iter2.current_rest_iter); 
    */
}  

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::reference
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator*() const
{
  return *current_rest_iter;
}

template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::pointer
FullArrayIterator<topleveliterT, restiterT, elemT, _Ref, _Ptr>::operator->() const
{
  return &(operator*());
}


END_NAMESPACE_TOMO
