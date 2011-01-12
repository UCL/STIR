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
#ifndef __stir_NestedIterator__H__
#define __stir_NestedIterator__H__

/*!
  \file 
 
  \brief This file declares the stir::NestedIterator class and supporting function objects.
  \ingroup buildblock

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/NestedIteratorHelpers.h"
#include "boost/iterator/iterator_traits.hpp"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief Class NestedIterator implements a (forward) iterator using 
  a pair of 'nested' iterators.

  Suppose you have a container where each element is a container, e.g. 
  <code>std::vector\<std::list\<int\> \> </code>. Using NestedIterator,
  you can iterate through the 'elements of the elements' (i.e. the
  int's in the above example).

  The template argument \c GetRestRangeFunctionT should be a function object
  that, given a top-level iterator, finds the first and last
  iterators for the sub-sequence. It defaults to just using
  \code
  current_rest_iter = top_level_iter->begin();
  end_rest_iter = top_level_iter->end();
  \endcode
  \see BeginEndFunction, PtrBeginEndFunction, ConstBeginEndFunction, ConstPtrBeginEndFunction,
  BeginEndAllFunction, PtrBeginEndAllFunction, ConstBeginEndAllFunction, ConstPtrBeginEndAllFunction

  Syntax is somewhat awkward for technical reasons (see the source for
  operator==). You have to give
  the \c begin and \c end of the top-level iterators at construction time.
  (This would be far more natural when using boost::range).
  
  \par examples
  Here is an example using a vector of lists of integers.
  \code
    typedef std::list<int> C2;
    typedef std::vector<C2> C1;
    C1 c;
    typedef NestedIterator<C1::iterator> FullIter;
    FullIter fiter(c.begin(),c.end());
    const FullIter fiter_end(c.end(),c.end());
    while (fiter != fiter_end)
    { ... }
  \endcode
  Here is an example using a vector of (smart) pointers to 2D arrays, where we want to
  iterate over all elements of all 2D arrays.
  \code
    typedef Array<2,int> C2;
    typedef std::vector<shared_ptr<C2> > C1;
    C1 c;
    typedef NestedIterator<C1::iterator, PtrBeginEndAllFunction<C1::iterator> > FullIter;
    FullIter fiter(c.begin(),c.end());
    const FullIter fiter_end(c.end(),c.end());
    while (fiter != fiter_end)
    { ... }
  \endcode
    
  \par Implementation note

  The 2nd template argument would really be better implemented
  as a template template. However, some compilers still don't support this.

  \bug At present, \c iterator_category typedef is hard-wired to be 
  \c std::forward_iterator_tag. This would be incorrect if
  \c topleveliterT or \c rest_iter_type is only an input or
  output iterator. 
*/
template <typename topleveliterT,
          class GetRestRangeFunctionT=BeginEndFunction<topleveliterT> >
class NestedIterator
{
private:
  typedef typename GetRestRangeFunctionT::rest_iter_type rest_iter_type;
public:
  typedef std::forward_iterator_tag iterator_category;
  typedef typename boost::iterator_difference<rest_iter_type>::type difference_type;
  typedef typename boost::iterator_value<rest_iter_type>::type value_type;
  typedef typename boost::iterator_reference<rest_iter_type>::type reference;
  typedef typename boost::iterator_pointer<rest_iter_type>::type pointer;  

public:
  //! default constructor
  inline NestedIterator();

  //! constructor to initialise the members
  inline NestedIterator(const topleveliterT& top_level_iter, 
                        const topleveliterT& end_top_level_iter);

  //! constructor to convert between nested iterators using convertible top and next level iterators
  /*! Ignore the 2nd and 3rd argument. They are there to let the compiler check if the types are 
      convertible (using the SFINAE principle).
      */
  template <typename othertopleveliterT, typename otherGetRestRangeFunctionT>
  inline NestedIterator(
			  NestedIterator<othertopleveliterT, otherGetRestRangeFunctionT> other,
			  typename boost::enable_if_convertible<othertopleveliterT, topleveliterT>::type* = 0,
			  typename boost::enable_if_convertible<typename otherGetRestRangeFunctionT::rest_iter_type, rest_iter_type>::type* = 0)
    : _current_top_level_iter(other._current_top_level_iter), 
      _end_top_level_iter(other._end_top_level_iter), 
      _current_rest_iter(other._current_rest_iter), 
      _end_rest_iter(other._end_rest_iter) 
    {}

  //inline NestedIterator& operator=(const NestedIterator&);
  
  //! prefix increment
  inline NestedIterator& operator++();

  //! postfix increment
  inline NestedIterator operator++(int);

  //! test equality
  inline bool operator==(const NestedIterator&) const;
  //! test equality
  inline bool operator!=(const NestedIterator&) const;
  
  //! dereferencing operator
  inline reference operator*() const;

  //! member-selection operator 
  inline pointer operator->() const;

#ifdef BOOST_NO_MEMBER_TEMPLATE_FRIENDS
public:
#else
private:   
  // needed for conversion constructor
  template <class,class> friend class NestedIterator;
#endif 

  //! the \c topleveliterT iterator pointing to the current \a row
  topleveliterT _current_top_level_iter;

  //! a \c topleveliterT iterator marking the end of the \a column
  topleveliterT _end_top_level_iter;

  //! a \c rest_iter_type iterator pointing to the current \a element in the \a row
  rest_iter_type _current_rest_iter;

  //! a \c rest_iter_type iterator pointing to the end of the current \a row
  rest_iter_type _end_rest_iter;
private:
  //! set the rest_iters to the range corresponding to a new \a top_level_iter
  void _set_rest_iters_for_current_top_level_iter();
};

END_NAMESPACE_STIR

#include "stir/NestedIterator.inl"

#endif
