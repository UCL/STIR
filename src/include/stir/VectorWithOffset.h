//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-01 - 2012, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_VectorWithOffset_H__
#define __stir_VectorWithOffset_H__

/*!
  \file 
  \ingroup Array  
  \brief defines the stir::VectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project

*/

#include "stir/common.h"
#include "boost/iterator/iterator_adaptor.hpp"
#include "boost/iterator/reverse_iterator.hpp"

START_NAMESPACE_STIR

namespace detail {
/*! \ingroup Array
  \brief templated class for the iterators used by VectorWithOffset.

  There should be no need to use this class yourself. Always use
  VectorWithOffset::iterator or VectorWithOffset::const_iterator.
*/
template <class elemT>
class VectorWithOffset_iter
  : public boost::iterator_adaptor<
       VectorWithOffset_iter<elemT>        // Derived
      , elemT*                             // Base
      , boost::use_default                 // Value
      , boost::random_access_traversal_tag // CategoryOrTraversal
    >
{
 private: 
  // abbreviation of the type of this class
  typedef VectorWithOffset_iter<elemT> self_t;
 public:
  VectorWithOffset_iter()
    : VectorWithOffset_iter::iterator_adaptor_(0) {}
  
  //! allow assignment from ordinary pointer
  /*! really should be used within VectorWithOffset
    It is explicit such that you can't do this by accident.
  */
  explicit VectorWithOffset_iter(elemT* p)
    : VectorWithOffset_iter::iterator_adaptor_(p) {}
  
  //! some magic trickery to be able to assign iterators to const iterators, but not to incompatible types
  /*! See the boost documentation for more info.
   */
  template <class OtherelemT>
    VectorWithOffset_iter(
			  VectorWithOffset_iter<OtherelemT> const& other,
			  typename boost::enable_if_convertible<OtherelemT, elemT>::type* = 0)
    : VectorWithOffset_iter::iterator_adaptor_(other.base()) {}
};

} // end of namespace detail


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

  It is possible to construct a VectorWithOffset that uses existing memory.
  It will then never deallocate that memory obviously. Note that when growing
  the vector (or assigning a bigger vector), the vector will allocate new 
  memory. Any modifications to the vector then will no longer be connected
  to the original data block. This can always be tested using
  owns_memory_for_data().

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
  /*! Most of these should really not be needed because we use boost::iterator_adaptor now.
      However, some are used directly in STIR code. (Maybe they shouldn't....)
  */
  //@{
  typedef T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef ptrdiff_t difference_type;
  typedef detail::VectorWithOffset_iter<T> iterator;
  typedef detail::VectorWithOffset_iter<T const> const_iterator;

  typedef boost::reverse_iterator<iterator> reverse_iterator;
  typedef boost::reverse_iterator<const_iterator> const_reverse_iterator;
  //@}
  typedef size_t size_type;
public:  
  

  //! Default constructor: creates a vector of length 0
  inline VectorWithOffset();
  
  //! Construct a VectorWithOffset of given length (initialised with \c T())
  inline explicit VectorWithOffset(const int hsz);
  
  //! Construct a VectorWithOffset with offset \c min_index (initialised with \c  T())
  inline VectorWithOffset(const int min_index, const int max_index);

  //! Construct a VectorWithOffset of given length using existing data (no initialisation)
  inline explicit 
    VectorWithOffset(const int hsz, 
		     T * const data_ptr,
		     T * const end_of_data_ptr);
  
  //! Construct a VectorWithOffset with offset \c min_index using existing data (no initialisation)
  inline 
    VectorWithOffset(const int min_index, const int max_index, 
		     T * const data_ptr,
		     T * const end_of_data_ptr);

  //! copy constructor
  inline VectorWithOffset(const VectorWithOffset &il) ;

  //! Destructor 
  inline virtual ~VectorWithOffset();	
  
  //! Free all memory and make object as if default-constructed
  /*! This is not the same as resize(0), as the latter does not 
    deallocate the memory (i.e. does not change the capacity()).
  */
  inline void recycle();

  //! assignment operator with another vector
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
  /*! New memory is allocated if the range grows outside the range specified by
      get_capacity_min_index() till get_capacity_max_index(). Data is then copied
      and old memory deallocated (unless owns_memory_for_data() is false).

      \todo in principle reallocation could be avoided when the new range would fit in the
      old one by shifting.
    */
  inline virtual void resize(const int min_index, const int max_index);

  //! change the range of the vector from 0 to new_size-1, new elements are set to \c T()
  inline void resize(const unsigned int new_size);


  //! make the allocated range at least from \a min_index to \a max_index
  inline void reserve(const int min_index, const int max_index);

  //! make the allocated range at least from 0 to new_size-1
  inline void reserve(const unsigned int new_size);

  //! get allocated size
  inline size_t capacity() const;

  //! check if this object owns the memory for the data
  /*! Will be false if one of the constructors is used that passes in a data block.
   */
  inline bool
    owns_memory_for_data() const;

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

  //! allow array-style access, read/write, but with range checking (throws std::out_of_range)
  inline T& at (int i);

  //! array access, read-only, but with range checking (throws std::out_of_range)
  inline const T& at(int i) const;
  
  //! checks if the vector is empty
  inline bool empty() const;

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

  inline reverse_iterator rbegin();
  inline reverse_iterator rend();
  inline const_reverse_iterator rbegin() const;
  inline const_reverse_iterator rend() const;
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
#if 0
  // next operators are disabled for now
  // if enabling then, you should also put them in NumericVectorWithOffset probably
  /*! \name arithmetic assignment operators with objects of the type \c T

     Note that currently the corresponding arithmetic operator are currently not implemented.
     One of the reasons is that the temporary copy it would involve is normally to
     be avoided at all costs.
   */
  //@{
  //! adding  \c t to the all elements of the current vector
  inline VectorWithOffset & operator+= (const T &t);

  //! subtracting \c t from the all elements of the current vector
  inline VectorWithOffset & operator-= (const T &t);

  //! multiplying elements of the current vector with \c t 
  inline VectorWithOffset & operator*= (const T &t);

  //! dividing all elements of the current vector by \c t
  inline VectorWithOffset & operator/= (const T &t);
  //@}
#endif

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
  bool _owns_memory_for_data;
};

END_NAMESPACE_STIR

#include "stir/VectorWithOffset.inl"

#endif // __VectorWithOffset_H__
