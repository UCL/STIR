//
// $Id$
//
/*
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
  \brief Definition of stir::is_null_ptr functions

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_is_null_ptr_H__
#define __stir_is_null_ptr_H__

#include "stir/shared_ptr.h"

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
{ return ptr==0; }

template <typename T>
inline 
bool 
is_null_ptr(shared_ptr<T> const& sptr)
{ return is_null_ptr(sptr.get()); }

#if 0
template <typename T>
inline 
bool 
is_null_ptr(auto_ptr<T> const& aptr)
{ return is_null_ptr(aptr.get()); }
#endif

//@}

END_NAMESPACE_STIR

#endif
