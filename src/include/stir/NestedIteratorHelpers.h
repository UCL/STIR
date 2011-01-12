//
// $Id$
//
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_NestedIteratorHelpers__H__
#define __stir_NestedIteratorHelpers__H__

/*!
  \file 
 
  \brief This file defines supporting function objects for stir::NestedIterator.
  \ingroup buildblock

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "stir/common.h"
#include "boost/iterator/iterator_traits.hpp"
#include "boost/pointee.hpp"


START_NAMESPACE_STIR

/*! \name Function objects that can be used for NestedIterator. 
  \ingroup buildblock
*/
//@{
#if 0
// This is the original implementation, which works but contains a lot of code repetition.
// It's superseded by the implementation below, but that might need a recent compiler, so
// I've kept this old version in the file for now, but it is disabled.
template <class TopLevelIterT>
class BeginEndFunction
{
  typedef typename boost::iterator_value<TopLevelIterT>::type iter_value_type;
 public:
  typedef typename iter_value_type::iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return iter->begin(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return iter->end(); }
};

template <class TopLevelIterT>
class ConstBeginEndFunction
{
  typedef typename boost::iterator_value<TopLevelIterT>::type iter_value_type;
 public:
  typedef typename iter_value_type::const_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return iter->begin(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return iter->end(); }
};

template <class TopLevelIterT>
class BeginEndAllFunction
{
  typedef typename boost::iterator_value<TopLevelIterT>::type iter_value_type;
 public:
  typedef typename iter_value_type::full_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return iter->begin_all(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return iter->end_all(); }
};

template <class TopLevelIterT>
class ConstBeginEndAllFunction
{
  typedef typename boost::iterator_value<TopLevelIterT>::type iter_value_type;
 public:
  typedef typename iter_value_type::const_full_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return iter->begin_all_const(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return iter->end_all_const(); }
};

template <class TopLevelIterT>
class PtrBeginEndFunction
{
  typedef typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type iter_value_type;
 public:
  typedef typename iter_value_type::iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return (**iter).begin(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return (**iter).end(); }
};

template <class TopLevelIterT>
class ConstPtrBeginEndFunction
{
  typedef typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type iter_value_type;
 public:
  typedef typename iter_value_type::const_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return (**iter).begin(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return (**iter).end(); }
};

template <class TopLevelIterT>
class PtrBeginEndAllFunction
{
  typedef typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type iter_value_type;
 public:
  typedef typename iter_value_type::full_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return (**iter).begin_all(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return (**iter).end_all(); }
};

template <class TopLevelIterT>
class ConstPtrBeginEndAllFunction
{
  typedef typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type iter_value_type;
 public:
  typedef typename iter_value_type::const_full_iterator rest_iter_type;

  rest_iter_type begin(const TopLevelIterT& iter) const
  {  return (**iter).begin_all_const(); }
  rest_iter_type end(const TopLevelIterT& iter) const
  {  return (**iter).end_all_const(); }
};

#else

// new implementation


//! Helper class for NestedIterator when the 1st level iterator refers to an ordinary iterator for the 2nd level iterator
/*! \param TopLevelIterT type of the top-level iterator to be used for NestedIterator.
    \param RestIterT type of the 2nd level iter.

    This class can be used for the case where the 2nd level iterators can be obtained by using \c top_level_iter.begin() etc,
    e.g. for \c std::vector<std::vector<int> >
    
    The default for \c RestIterT is suitable for most cases, as it finds \c RestIterT from the iterator traits
    of \c TopLevelIterT.
*/
template <class TopLevelIterT, class RestIterT = typename boost::iterator_value<TopLevelIterT>::type::iterator>
class BeginEndFunction
{
public:
  //! typedef storing the type of the 2nd level iterator
  typedef RestIterT rest_iter_type;
  //! function to get the first 2nd level iterator for a top-level iterator
  /*! returns \c (*iter).begin() */
  inline RestIterT begin(const TopLevelIterT& iter) const
  {  return iter->begin(); }
  //! function to get the "end" 2nd level iterator for a top-level iterator
  inline RestIterT end(const TopLevelIterT& iter) const
  {  return iter->end(); }
};

//! Helper class for NestedIterator when the 1st level iterator refers to pointers to an ordinary iterator for the 2nd level iterator
/*! \param TopLevelIterT type of the top-level iterator to be used for NestedIterator.
    \param RestIterT type of the 2nd level iter.

    This class can be used for the case where the 2nd level iterators can be obtained by using \c top_level_iter->begin() etc,
    e.g. for \c std::vector<std::vector<int> *> or std::vector<shared_ptr<std::vector<int> > >. 

    The default for \c RestIterT is suitable for most cases, as it finds \c RestIterT from the iterator traits
    of \c TopLevelIterT. 

    \see BeginEndFunction
*/
template <class TopLevelIterT, 
          class RestIterT = typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type::iterator>
class PtrBeginEndFunction
{
public:
  typedef RestIterT rest_iter_type;
  //! function to get the first 2nd level iterator for a top-level iterator
  /*! returns \c (**iter).begin() */
  inline RestIterT begin(const TopLevelIterT& iter) const
  {   return (**iter).begin(); }
  //! function to get the "end" 2nd level iterator for a top-level iterator
  inline RestIterT end(const TopLevelIterT& iter) const
  {  return (**iter).end(); }
};

//! Helper class for NestedIterator when the 1st level iterator refers to a stir full iterator for the 2nd level iterator
/*! \param TopLevelIterT type of the top-level iterator to be used for NestedIterator.
    \param RestIterT type of the 2nd level iter.

    This class can be used for the case where the 2nd level iterators can be obtained by using \c top_level_iter.begin_all() etc,
    e.g. for \c std::vector<Array<2,float> >
    
    The default for \c RestIterT is suitable for most cases, as it finds \c RestIterT from the iterator traits
    of \c TopLevelIterT.

    \see BeginEndFunction
*/
template <class TopLevelIterT, class RestIterT = typename boost::iterator_value<TopLevelIterT>::type::full_iterator>
class BeginEndAllFunction
{
public:
  typedef RestIterT rest_iter_type;
  inline RestIterT begin(const TopLevelIterT& iter) const
  {  return iter->begin_all(); }
  inline RestIterT end(const TopLevelIterT& iter) const
  {  return iter->end_all(); }
};

//! Helper class for NestedIterator when the 1st level iterator refers to a pointer to a stir full iterator for the 2nd level iterator
/*! \param TopLevelIterT type of the top-level iterator to be used for NestedIterator.
    \param RestIterT type of the 2nd level iter.

    This class can be used for the case where the 2nd level iterators can be obtained by using \c top_level_iter->begin_all() etc,
    e.g. for \c std::vector<Array<2,float> * > or std::vector<shared_ptr<Array<2,float> > >.
    
    The default for \c RestIterT is suitable for most cases, as it finds \c RestIterT from the iterator traits
    of \c TopLevelIterT.

    \see BeginEndFunction, BeginEndAllFunction
*/
template <class TopLevelIterT, 
          class RestIterT = typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type::full_iterator>
class PtrBeginEndAllFunction
{
public:
  typedef RestIterT rest_iter_type;
  inline RestIterT begin(const TopLevelIterT& iter) const
  {  return (**iter).begin_all(); }
  inline RestIterT end(const TopLevelIterT& iter) const
  {  return (**iter).end_all(); }
};

//! Convenience class where the 2nd level iterator is a \c const_iterator
/*! This class just changes the default of BeginEndFunction. */
template <class TopLevelIterT, class RestIterT = typename boost::iterator_value<TopLevelIterT>::type::const_iterator>
class ConstBeginEndFunction 
  : public BeginEndFunction<TopLevelIterT, RestIterT>
{};

//! Convenience class where the 2nd level iterator is a \c const_full_iterator
/*! This class just changes the default of BeginEndAllFunction. */
template <class TopLevelIterT, class RestIterT = typename boost::iterator_value<TopLevelIterT>::type::const_full_iterator>
class ConstBeginEndAllFunction 
  : public BeginEndAllFunction<TopLevelIterT, RestIterT>
{};

//! Convenience class where the 2nd level iterator is a \c const_iterator
/*! This class just changes the default of PtrBeginEndFunction. */
template <class TopLevelIterT>
class ConstPtrBeginEndFunction 
  : public PtrBeginEndFunction<TopLevelIterT, 
                               typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type::const_iterator>
{};

//! Convenience class where the 2nd level iterator is a \c const_full_iterator
template <class TopLevelIterT>
class ConstPtrBeginEndAllFunction
  : public PtrBeginEndAllFunction<TopLevelIterT, 
                                  typename boost::pointee<typename boost::iterator_value<TopLevelIterT>::type>::type::const_full_iterator>
{};

#endif // #if between old and new implementation

//@}


END_NAMESPACE_STIR

#endif
