//
// $Id$
//
// This file is for doxygen only. It does not contain any code.
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

/*! \namespace boost
  \brief Namespace for the boost library 

  see http://www.boost.org
*/

namespace boost
{
//! Boost class to define all comparison operators given only 2
template <class T, class B = ::boost::detail::empty_base>
  struct partially_ordered;

//! Boost class to define operator!= in terms of operator==
template <class T, class B = ::boost::detail::empty_base>
  struct equality_comparable;

}

