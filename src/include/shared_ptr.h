//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of class shared_ptr
    
  \author Mustapha Sadki (minor modifications by Kris Thielemans)
  \author PARAPET project
      
  $Date$        
  $Revision$
*/         

#ifndef __Tomo_SHARED_PTR__
#define __Tomo_SHARED_PTR__

#include "tomo/common.h"

#ifdef TOMO_USE_BOOST
#include "boost/smart_ptr.hpp"
START_NAMESPACE_TOMO
using boost::shared_ptr;
END_NAMESPACE_TOMO

#else
// no boost implementation
#include <memory>
#ifndef TOMO_NO_NAMESPACES
#ifndef TOMO_NO_AUTO_PTR
using std::auto_ptr;
#endif
#endif

START_NAMESPACE_TOMO


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
  
#ifndef TOMO_NO_AUTO_PTR
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

#endif // TOMO_USE_BOOST

END_NAMESPACE_TOMO

#include "shared_ptr.inl"

#endif
