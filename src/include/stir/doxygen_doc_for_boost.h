//
// $Id$
//
//#error This file is for doxygen only. It does not contain any code.
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

/*! \namespace boost
  \brief Namespace for the boost library 

  see http://www.boost.org
*/

/*! \defgroup boost Boost

  Pending doxygen documentation of the boost library, we provide very basic
  documentation of a few boost classes within STIR. This is mainly to get
  class hierarchies and so on to work.

  See http://www.boost.org for more info.
*/

namespace boost
{
  namespace detail {
  /*! \ingroup boost
  \brief Boost class for chaining of operators (see operators.hpp)
  */
    class empty_base {};
  }

/*! \ingroup boost
  \brief Boost class to define all comparison operators given only 2 (see operators.hpp)
  */
template <class T, class B = ::boost::detail::empty_base>
  struct partially_ordered  {};

/*! \ingroup boost
  \brief Boost class to define operator!= in terms of operator== (see operators.hpp)
  */
template <class T, class B = ::boost::detail::empty_base>
  struct equality_comparable {};

}

