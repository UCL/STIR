//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Definition of is_null_ptr functions

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
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
bool 
is_null_ptr(shared_ptr<T> const& sptr)
{ return is_null_ptr(sptr.get()); }

#if 0
template <typename T>
bool 
is_null_ptr(auto_ptr<T> const& aptr)
{ return is_null_ptr(aptr.get()); }
#endif

template <typename T>
bool 
is_null_ptr(T const * const ptr)
{ return ptr==0; }

//@}

END_NAMESPACE_STIR

#endif
