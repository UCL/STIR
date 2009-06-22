// $Id$
/*!
  \file 
  \ingroup buildblock_detail 
  \brief Classes for use in implementation of stir::Array, stir::BasicCoordinate etc to test if it's a 1D array.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_detail_test_if_1d_H__
#define __stir_detail_test_if_1d_H__
namespace stir {
  namespace detail {
    /*! \ingroup buildblock_detail
       \brief a class used to signify it's a 1D array
       \see test_if_1d
    */
    struct is_1d{};
    /*! \ingroup buildblock_detail
       \brief a class used to signify it's not a 1D array
       \see test_if_1d
    */
    struct is_not_1d{};

    /*! \ingroup buildblock_detail
       \brief a templated class used to check if it's a 1D array or not
       This class only exists to allow a work-around for older compilers 
       (such as VC 6.0) that do not implement partial ordering of
       function templates or partial template specialisation.

       For modern compilers one can write
       \code
       // generic case
       template <int n, class T> void f(Array<n,T>&);
       // 1D case
       template <class T>        void f(Array<1,T>&);
       \endcode
       The work-around is as follows
       \code
       // generic case
       template <int n, class T> void f_help(is_not_1d, Array<n,T>&);
       // 1D case
       template <class T>        void f_help(is_1d, Array<1,T>&);
       // function that will dispatch
       template <int n, class T> void f(Array<n,T>& a)
       { f_help(test_if_1d<n>(), a); }
       \endcode
       So, the same effect is achieved by having one extra function.
       Of course, the name <tt>f_help</tt> is arbitrary, and could just as well
       be <tt>f</tt>. However, it's best to hide these away from the user, as they
       should never be used explicitly.
    */
    // note: should be Num_dimensions and not num_dimensions 
    // because for old compilers, num_dimensions is sometimes #defined (sigh)
    template <int Num_dimensions>
      struct test_if_1d : is_not_1d {};
    /*! \ingroup buildblock_detail
       \brief 1D specialisation of a templated class used to check if it's a 1D array or not
    */
    template <>
      struct test_if_1d<1> : is_1d {};
  }
}

#endif
