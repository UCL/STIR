//
//
//#error This file is for doxygen only. It does not contain any code.
/*!
  \file
  \ingroup boost
  
  \brief Documentation for some boost functions
    
  \author Kris Thielemans
  \author PARAPET project
      
        
*/         

/*
    Copyright (C) 2000 PARAPET project
    Copyright (C) 2001 - 2004-09-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans

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


/*! \ingroup boost
  \brief A smart pointer class: multiple shared_ptr's refer to one object

  This class keeps a reference counter to see how many shared_ptr's refer
  to the object. When a shared_ptr is deleted, the reference counter is 
  decremented and if the object is longer referenced, it is deleted.

  \par Advantages: (it's easy)

  <ul>
  <li> Automatic tracking of memory allocations. No memory leaks.
  <li> Syntax hardly changes (you still use * and ->)
  </ul>

  \par Disadvantages: (you have to be careful)
  <ul>
  <li> If the object which a shared_ptr refers to gets modified, it affects all 
  shared_ptrs sharing the object.
  <li> Constructing 2 shared_ptr's from the same ordinary pointer gives trouble.
  </ul>

  \par Example:

  \code
  
  { 
    // ok
    shared_ptr<int> i_ptr1(new int (2));
    shared_ptr<int> i_ptr2(i_ptr1);
    unique_ptr<int> a_ptr(new int(3));
    shared_ptr<int> i_ptr3(a_ptr);
    // now never use a_ptr anymore
    {
      int * i_ptr = new int (2);
      i_ptr1.reset(i_ptr);
      // now never use i_ptr anymore
    }
  }
  { 
    // trouble! *i_ptr will be deleted twice !
    int * i_ptr = new int (2);
    shared_ptr<int> i_ptr1 (i_ptr);
    shared_ptr<int> i_ptr2 (i_ptr);
  }
  \endcode
*/
template <class T>
  class shared_ptr {};


}
