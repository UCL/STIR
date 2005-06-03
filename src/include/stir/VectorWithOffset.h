//
// $Id$
//
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
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_VectorWithOffset_H__
#define __stir_VectorWithOffset_H__

/*!
  \file 
  \ingroup Array  
  \brief defines the VectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/common.h"

#include <iterator>
#ifndef STIR_NO_NAMESPACES
using std::size_t;
using std::ptrdiff_t;
using std::random_access_iterator_tag;
#endif

START_NAMESPACE_STIR


/*! 
  \ingroup Array
  \brief A templated class for vectors, but with indices starting not from 0

  Elements are guaranteed to be stored contiguously. (Emergency) methods
  are provided for accessing the data via a \c T*.

  This class tries to mimic std::vector for the most common methods, but
  it is much more conservative in its memory allocations. 
  The only memory that is allocated is what you asked for (although the
  allocated memory hardly ever shrinks, except when calling recycle()).
  So, std::vector::push_back() etc are not provided, as they would be
  horribly inefficient for the current class (ok, except if you would have
  called reserve() first).

  \warning This class does not satisfy full Container requirements.
  \warning Current implementation relies on shifting a \c T* outside the range
  of allocated data. This is not guaranteed to be valid by ANSI C++. It is fine
  however as long as the \c min_index is negative and such that \c abs(min_index) is
  smaller than \c max_index.
  \todo add allocator template as in std::vector. This is non-trivial as
  we would have to use uninitialized_copy etc. in some places.
*/

template <class T>
class VectorWithOffset
{
public:
  //! \name typedefs for iterator support
  /*! \todo set iterator_traits */
  //@{
  typedef random_access_iterator_tag iterator_category;  
  typedef T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
  //@}
public:  
  

  //! Default constructor: creates a vector of length 0
  inline VectorWithOffset();
  
  //! Construct a VectorWithOffset of given length (initialised with \c T())
  inline explicit VectorWithOffset(const int hsz);
  
  //! Construct a VectorWithOffset with offset \c min_index (initialised with \c  T())
  inline VectorWithOffset(const int min_index, const int max_index);

  //! copy constructor
  inline VectorWithOffset(const VectorWithOffset &il) ;

  //! Destructor 
  inline virtual ~VectorWithOffset();	
  
  //! Free all memory and make object as if default-constructed
  /*! This is not the same as resize(0), as the latter does not 
    deallocate the memory (i.e. does not change the capacity()).
  */
  inline void recycle();

  //! assignment operator
  inline VectorWithOffset & operator= (const VectorWithOffset &il) ;

  //! \name index range operations
  //@{ 
  //! return number of elements in this vector
  /*! \deprecated Use size() instead. */
  inline int get_length() const;	
  
  //! return number of elements in this vector
  inline size_t size() const;	

  //! get value of first valid index
  inline int get_min_index() const;

  //! get value of last valid index
  inline int get_max_index() const;

  //! change value of starting index
  inline void set_offset(const int min_index);
  
  //! identical to set_offset()
  inline void set_min_index(const int min_index);

  //! grow the range of the vector, new elements are set to \c T()
  /*! Currently, it is only checked with assert() if old range is 
      a subinterval of the new range.

      grow() currently simply calls resize(). However, if you overload
      resize() in a derived class, it is probably safest to overload 
      grow() as well.
  */
  inline virtual void grow(const int min_index, const int max_index);

  //! grow the range of the vector from 0 to new_size-1, new elements are set to \c T()
  inline void grow(const unsigned int new_size);

  //! change the range of the vector, new elements are set to \c T()
  inline virtual void resize(const int min_index, const int max_index);

  //! change the range of the vector from 0 to new_size-1, new elements are set to \c T()
  inline void resize(const unsigned int new_size);


  //! make the allocated range at least from \a min_index to \a max_index
  inline void reserve(const int min_index, const int max_index);

  //! make the allocated range at least from 0 to new_size-1
  inline void reserve(const unsigned int new_size);

  //! get allocated size
  inline size_t capacity() const;

  //! get min_index within allocated range
  /*! This value depends on get_min_index() and hence will change 
      after calling set_min_index(). 

      For a vector of 0 length, this function returns 0.
  */
  inline int get_capacity_min_index() const;

  //! get max_index within allocated range
  /*! This value depends on get_min_index() and hence will change 
      after calling set_min_index().

      For a vector of 0 length, this function returns capacity()-1.
  */
  inline int get_capacity_max_index() const;
  //@}
  
  //! allow array-style access, read/write
  inline T& operator[] (int i);

  //! array access, read-only
  inline const T& operator[] (int i) const;
  
  //! \name comparison operators
  //@{
  inline bool operator== (const VectorWithOffset &iv) const;
  inline bool operator!= (const VectorWithOffset &iv) const;
  //@}

  //! fill elements with value \a n
  inline void fill(const T &n);

  //! \name access to the data via a pointer
  //@{
  //! member function for access to the data via a T*
  inline T* get_data_ptr();

  //! member function for access to the data via a const T*
#ifndef STIR_NO_MUTABLE
  inline const T * get_const_data_ptr() const;
#else
  inline const T * get_const_data_ptr();
#endif

  //! signal end of access to T*
  inline void release_data_ptr();

  //! signal end of access to const T*
#ifndef STIR_NO_MUTABLE
  inline void release_const_data_ptr() const;
#else
  inline void release_const_data_ptr();
#endif
  //@}

  //!\name basic iterator support
  //@{
  //! use to initialise an iterator to the first element of the vector
  inline iterator begin();
  //! use to initialise an iterator to the first element of the (const) vector
  inline const_iterator begin() const;
  //! iterator 'past' the last element of the vector
  inline iterator end();
  //! iterator 'past' the last element of the (const) vector
  inline const_iterator end() const;
  //@}

  /*! \name arithmetic assignment operators with objects of the same type
      \warning Arguments must have matching index ranges. Otherwise error() is called. 
   */
  //@{
  //! adding elements of \c v to the current vector
  inline VectorWithOffset & operator+= (const VectorWithOffset &v);

  //! subtracting elements of \c v from the current vector
  inline VectorWithOffset & operator-= (const VectorWithOffset &v);

  //! multiplying elements of the current vector with elements of \c v 
  inline VectorWithOffset & operator*= (const VectorWithOffset &v);

  //! dividing all elements of the current vector by elements of \c v
  inline VectorWithOffset & operator/= (const VectorWithOffset &v);
  //@}
  /*! \name arithmetic operators with objects of the same type
   
     \warning Arguments must have matching index ranges. Otherwise error() is called. 
     \warning current implementation involves a temporary copy of the data,
        unless you have a really smart compiler.
   */
  //@{
  //! adding vectors, element by element
  inline VectorWithOffset operator+ (const VectorWithOffset &v) const;

  //! subtracting vectors, element by element
  inline VectorWithOffset operator- (const VectorWithOffset &v) const;

  //! multiplying vectors, element by element
  inline VectorWithOffset operator* (const VectorWithOffset &v) const;

  //! dividing vectors, element by element
  inline VectorWithOffset operator/ (const VectorWithOffset &v) const;
  //@}
 
protected:
  
  //! pointer to (*this)[0] (taking get_min_index() into account that is).
  T *num;	
  
  //! Called internally to see if all variables are consistent
  inline void check_state() const;

private:
  //! length of vector
  unsigned int length;	
  //! starting index
  int start;	

  T* begin_allocated_memory;
  T* end_allocated_memory;

  //! Default member settings for all constructors
  inline void init();

  //! call destructors and deallocate
  inline void _destruct_and_deallocate();
  
  //! boolean to test if get_data_ptr is called
  // This variable is declared mutable such that get_const_data_ptr() can change it.
#ifndef STIR_NO_MUTABLE
  mutable
#endif
  bool pointer_access;

};

END_NAMESPACE_STIR

#include "stir/VectorWithOffset.inl"

#endif // __VectorWithOffset_H__
