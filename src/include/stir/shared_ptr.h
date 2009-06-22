//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of class stir::shared_ptr
    
  \author Mustapha Sadki (minor modifications by Kris Thielemans)
  \author PARAPET project
      
  $Date$        
  $Revision$
*/         
/*
    Copyright (C) 2000 PARAPET partners
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

#ifndef __stir_SHARED_PTR__
#define __stir_SHARED_PTR__

#include "stir/common.h"

#ifdef STIR_USE_BOOST
#include "boost/smart_ptr.hpp"
START_NAMESPACE_STIR
using boost::shared_ptr;
END_NAMESPACE_STIR

#else
// no boost implementation
#include <memory>
#ifndef STIR_NO_NAMESPACES
#ifndef STIR_NO_AUTO_PTR
using std::auto_ptr;
#endif
#endif

START_NAMESPACE_STIR


/*!
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
    shared_ptr<int> i_ptr1 = new int (2);
    shared_ptr<int> i_ptr2 = i_ptr1;
  }
  { 
    // trouble! *i_ptr will be deleted twice !
    int * i_ptr = new int (2);
    shared_ptr<int> i_ptr1 = i_ptr;
    shared_ptr<int> i_ptr2 = i_ptr;
  }
  \endcode
*/
template <class T>
class shared_ptr 
{
public:    
  inline shared_ptr(T * t = NULL);
  
  inline ~shared_ptr();
  
  inline shared_ptr(const shared_ptr<T> & cp);
  
#ifndef STIR_NO_AUTO_PTR
  inline shared_ptr(auto_ptr<T>& r);
#endif
  
  inline shared_ptr<T> &
    operator= (const shared_ptr<T> & cp);
  
  // TODO add assignment from auto_ptr
  
  inline T * operator-> () const;
  
  inline T &  operator* () const;

#if 0
  template <class newType>
    inline shared_ptr<newType> operator();
#endif
  
  inline bool  operator== (const shared_ptr<T> & cp) const;
  inline bool  operator!= (const shared_ptr<T> & cp) const;
  
  // KT 10/05/2000 removed as not in boost::shared_ptr
#if 0
  inline void  release();
  
  inline bool is_null() const;
#endif
  
  inline T* get() const;
  
  inline long use_count() const;
  
private:
  struct couple 
  {
    T *               data;
    // KT 10/05/2000 removed unsigned for boost::shared_ptr compatibility
    long     count;       
    couple(T * t): data(t), count(1) {}
    ~couple() { 
      assert(data != 0);
      delete data; 
#ifndef NDEBUG
      data = 0;
#endif
    }
  };    
  struct couple *    ptr;
  
};

#endif // STIR_USE_BOOST

END_NAMESPACE_STIR

#include "stir/shared_ptr.inl"

#endif
