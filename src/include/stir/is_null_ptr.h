//
//
/*
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2016, University College London
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
  \brief Definition of stir::is_null_ptr functions

  \author Kris Thielemans
*/

#ifndef __stir_is_null_ptr_H__
#define __stir_is_null_ptr_H__

#include "stir/shared_ptr.h"
#include <memory>
#include "stir/unique_ptr.h"
START_NAMESPACE_STIR
/*! \ingroup buildblock
  \name testing of (smart) pointers
  
  A utility function that checks if an ordinary or smart pointer is null
  with identical syntax for all types.
*/
//@{
template <typename T>
inline 
bool 
is_null_ptr(T const * const ptr)
{
#ifdef BOOST_NO_CXX11_NULLPTR
  return ptr==0;
#else
  return ptr==nullptr;
#endif
}

template <typename T>
inline 
bool 
is_null_ptr(shared_ptr<T> const& sptr)
{ return is_null_ptr(sptr.get()); }

template <typename T>
inline 
bool 
is_null_ptr(unique_ptr<T> const& aptr)
{ return is_null_ptr(aptr.get()); }

#ifndef BOOST_NO_CXX11_NULLPTR
inline 
bool 
is_null_ptr(const std::nullptr_t)
{
  return true;
}
#endif

//@}

END_NAMESPACE_STIR

#endif
