//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup buildblock 
  \brief inline implementations of VectorWithOffset

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/


#include <algorithm>

START_NAMESPACE_TOMO

template <class T>
void 
VectorWithOffset<T>::init() 
{		
  length =0;	// i.e. an empty row of zero length,
  start = 0;	// no offsets
  num = 0;	// and no data.
}

/*!
This function (only non-empty when debugging)
is used before and after any modification of the object
*/
template <class T>
void 
VectorWithOffset<T>::check_state() const
{
  // disable for normal debugging
#if _DEBUG>1
  assert(((length > 0) ||
	  (length == 0 && start == 0 &&
	   num == 0)));
  
#endif
  // check if data is being access via a pointer (see get_data_ptr())
  assert(pointer_access == false);
}

template <class T>
void 
VectorWithOffset<T>::Recycle() 
{
  check_state();
  if (length > 0)
  {
    delete[] begin(); 
    init();
  }
}

template <class T>
int VectorWithOffset<T>::get_min_index() const 
{ 
  return start; 
}

template <class T>
int VectorWithOffset<T>::get_max_index() const 
{ 
  return start + length - 1; 
}

/*! Out of range errors are detected using assert() */
template <class T>
T& 
VectorWithOffset<T>::operator[] (int i) 
{
  check_state();
  assert(i>=get_min_index());
  assert(i<=get_max_index());
  
  return num[i];
}

/*! Out of range errors are detected using assert() */
template <class T>
const T& 
VectorWithOffset<T>::operator[] (int i) const 
{ 
  check_state();
  assert(i>=get_min_index());
  assert(i<=get_max_index());
  
  return num[i];
}

template <class T>
VectorWithOffset<T>::iterator 
VectorWithOffset<T>::begin() 
{ 
  check_state();
  return num+get_min_index(); 
}

template <class T>
VectorWithOffset<T>::const_iterator 
VectorWithOffset<T>::begin() const 
{
  check_state();
  return num+get_min_index(); 
}

template <class T>
VectorWithOffset<T>::iterator 
VectorWithOffset<T>::end() 
{
  check_state();
  return num+get_max_index()+1; 
}

template <class T>
VectorWithOffset<T>::const_iterator 
VectorWithOffset<T>::end() const 
{ 
  check_state();
  return num+get_max_index()+1; 
}

template <class T>

VectorWithOffset<T>::VectorWithOffset()
{ 
  pointer_access = false;  
  init();
}

template <class T>
VectorWithOffset<T>::VectorWithOffset(const int hsz)
  : length(hsz),
    start(0),
    pointer_access(false)
{	
  if ((hsz > 0))
  {
    num = new T[hsz];
  }
  else 
    init();
  check_state();
}			

template <class T>

VectorWithOffset<T>::VectorWithOffset(const int min_index, const int max_index)   
  : length(max_index - min_index + 1),
    start(min_index),
    pointer_access(false)
{   
  if (length > 0) 
  {
    num = new T[length];
    num -= min_index;
  } 
  else 
    init(); 
  check_state();
}

template <class T>

VectorWithOffset<T>::~VectorWithOffset()
{ 
  Recycle(); 
}		

template <class T>
void 
VectorWithOffset<T>::set_offset(const int min_index) 
{
  check_state();
  //  only allowed when non-zero length
  if (length == 0) return;  
  num += start - min_index;
  start = min_index;
}

//the new members will be initialised with the default constructor for T
template <class T>
void 
VectorWithOffset<T>::grow(const int min_index, const int max_index) 
{ 
  check_state();
  const int new_length = max_index - min_index + 1;
  if (min_index == start && new_length == length) {
    return;
  }
  
  // allow grow arbitrary when it's zero length
  assert(length == 0 || (min_index <= start && new_length >= length));
  T *newnum = new T[new_length];
  newnum -= min_index;
  // TODO hopefully replace by using when not using pet_common.h
  std::copy(begin(), end(), newnum+start);
  // Check on length probably not really necessary
  // as begin() is == 0 in that case, and delete[] 0 doesn't do anything
  // (I think). Anyway, we're on the safe side now...
  if (length != 0)
    delete [] (begin());
  num = newnum;
  length = new_length;
  start = min_index;
  check_state();
}

template <class T>
VectorWithOffset<T> & 
VectorWithOffset<T>::operator= (const VectorWithOffset &il) 
{
  check_state();
  if (this == &il) return *this;		// in case of x=x
  if (il.length == 0)
  {
    Recycle();
  }
  else
  {		
    if (length != il.length)
    {		
      // if new VectorWithOffset has different length, reallocate memory
      // KT 31/01/2000 did optimisation
      //in fact, the test on length can be skipped, because when
      //length == 0, mem == 0, and delete [] 0 doesn't do anything
      //???check
      if (length > 0) delete [] begin();
      // Recycle();
      length = il.length;
      num = new T[length];
      // set such that set_offset() below works
      start = 0;
    }
    set_offset(il.get_min_index());

    std::copy(il.begin(), il.end(), begin());
  }
  
    
  check_state();
  return *this;
}


template <class T>
VectorWithOffset<T>::VectorWithOffset(const VectorWithOffset &il) 
{
  pointer_access = false;
  
  init();
  *this = il;		// Uses assignment operator (above)
}

template <class T>
int VectorWithOffset<T>::get_length() const 
{ 
  check_state(); 
  return length; 
}

template <class T>
bool 
VectorWithOffset<T>::operator== (const VectorWithOffset &iv) const
{
  check_state();
  if (length != iv.length || start != iv.start) return false;
  return equal(begin(), end(), iv.begin());
}

template <class T>
void 
VectorWithOffset<T>::fill(const T &n) 
{
  check_state();
  //TODO use std::fill() if we can use namespaces (to avoid name conflicts)
  //std::fill(begin(), end(), n);
  for(int i=get_min_index(); i<=get_max_index(); i++)
    num[i] = n;
  check_state();
}



/*! 
  This returns a \c T* to the first element of a, 
  members are guaranteed to be stored contiguously in memory.

  Use only in emergency cases...

  To prevent invalidating the safety checks (and making 
  reimplementation more difficult), NO manipulation with
  the vector is allowed between the pairs
      get_data_ptr() and release_data_ptr()
  and
      get_const_data_ptr() and release_data_ptr().
  (This is checked with assert() in DEBUG mode.)
*/
template <class T>
T* 
VectorWithOffset<T>::get_data_ptr()
{
  assert(!pointer_access);
  
  pointer_access = true;
  return (num+start);
  
  // if implementation changes, this would need to keep track 
  // if which pointer it returns.
};

/*! 
  This returns a \c const \c T* to the first element of a, 
  members are guaranteed to be stored contiguously in memory.

  Use get_const_data_ptr() when you are not going to modify
  the data.

  \see get_data_ptr()
*/
template <class T>
const T *  
VectorWithOffset<T>::get_const_data_ptr()
#ifndef TOMO_NO_MUTABLE
const
#endif
{
  assert(!pointer_access);
  
  pointer_access = true;
  return (num+start);
  
  // if implementation changes, this would need to keep track 
  // if which pointer it returns.
};

/*! 
  This has to be used when access to the T* is finished. It updates
  the vector with any changes you made, and allows access to 
  the other member functions again.

  \see get_data_ptr()
*/
template <class T>
void 
VectorWithOffset<T>::release_data_ptr()
{
  assert(pointer_access);
  
  pointer_access = false;
}

END_NAMESPACE_TOMO
