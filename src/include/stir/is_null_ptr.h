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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_is_null_ptr_H__
#define __stir_is_null_ptr_H__

#include "stir/shared_ptr.h"

START_NAMESPACE_STIR
//! a utility function that checks if a shared_ptr is null
template <typename T>
bool 
is_null_ptr(shared_ptr<T> const& sptr)
{ return is_null_ptr(sptr.get()); }

#if 0
// TODO check before enable
//! a utility function that checks if an auto_ptr is null
template <typename T>
bool 
is_null_ptr(auto_ptr<T> const& aptr)
{ return is_null_ptr(aptr.get()); }
#endif

//! a utility function that checks if an ordinary_ptr is null
template <typename T>
bool 
is_null_ptr(T const * const ptr)
{ return ptr==0; }

END_NAMESPACE_STIR

#endif
