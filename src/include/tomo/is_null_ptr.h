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

#ifndef __Tomo_is_null_ptr_H__
#define __Tomo_is_null_ptr_H__

#include "shared_ptr.h"

START_NAMESPACE_TOMO
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

END_NAMESPACE_TOMO

#endif
