
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-10-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2023 - 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Array_H__
#define __stir_Array_H__

#ifndef ARRAY_FULL
#  define ARRAY_FULL
#endif

/*!
  \file
  \ingroup Array
  \brief defines the stir::Array class for multi-dimensional (numeric) arrays

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project
  \author Gemma Fardell

*/
#include "stir/NumericVectorWithOffset.h"
#include "stir/ByteOrder.h"
#include "stir/IndexRange.h"
#include "stir/deprecated.h"
#include "stir/shared_ptr.h"
// include forward declaration to ensure consistency, as well as use of
// default parameters (if any)
#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR
class NumericType;

#ifdef ARRAY_FULL
#  ifndef ARRAY_FULL2
template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
class FullArrayIterator;
#  else
template <int num_dimensions, typename elemT, typename _Ref, typename _Ptr>
class FullArrayIterator;
template <int num_dimensions, typename elemT, typename _Ref, typename _Ptr>
class FullArrayConstIterator;
#  endif

#endif

/*!
  \ingroup Array
  \brief This class defines multi-dimensional (numeric) arrays.

  This class implements multi-dimensional arrays which can have
'irregular' ranges. See IndexRange for a description of the ranges. Normal
numeric operations are defined. In addition, two types of iterators are
defined, one which iterators through the outer index, and one which
iterates through all elements of the array.

Array inherits its numeric operators from NumericVectorWithOffset.
In particular this means that operator+= etc. potentially grow
the object. However, as grow() is a virtual function, Array::grow is
called, which initialises new elements first to 0.
*/

template <int num_dimensions, typename elemT>
class Array : public NumericVectorWithOffset<Array<num_dimensions - 1, elemT>, elemT>
{
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
private:
#endif
  typedef Array<num_dimensions, elemT> self;
  typedef NumericVectorWithOffset<Array<num_dimensions - 1, elemT>, elemT> base_type;

public:
  //@{
  //! typedefs such that we do not need to have \c typename wherever we use these types defined in the base class
  typedef typename base_type::value_type value_type;
  typedef typename base_type::reference reference;
  typedef typename base_type::const_reference const_reference;
  typedef typename base_type::difference_type difference_type;
  typedef typename base_type::size_type size_type;
  typedef typename base_type::iterator iterator;
  typedef typename base_type::const_iterator const_iterator;
  //@}
#ifdef ARRAY_FULL
  /*! @name typedefs for full_iterator support
  Full iterators provide a 1-dimensional view on a multi-dimensional
  Array.
  */

  //@{
  typedef elemT full_value_type;
  typedef full_value_type* full_pointer;
  typedef const full_value_type* const_full_pointer;
  typedef full_value_type& full_reference;
  typedef const full_value_type& const_full_reference;
#  ifndef ARRAY_FULL2
  //! This defines an iterator type that iterates through all elements.
  typedef FullArrayIterator<typename base_type::iterator,
                            typename Array<num_dimensions - 1, elemT>::full_iterator,
                            elemT,
                            full_reference,
                            full_pointer>
      full_iterator;

  //! As full_iterator, but for const objects.
  typedef FullArrayIterator<typename base_type::const_iterator,
                            typename Array<num_dimensions - 1, elemT>::const_full_iterator,
                            elemT,
                            const_full_reference,
                            const_full_pointer>
      const_full_iterator;
#  else // ARRAY_FULL2
  typedef FullArrayIterator<num_dimensions, elemT, full_reference, full_pointer> full_iterator;

  typedef FullArrayConstIterator<num_dimensions, elemT, const_full_reference, const_full_pointer> const_full_iterator;

#  endif
  //@}
#endif
public:
  //! Construct an empty Array
  inline Array();

  //! Construct an Array of given range of indices, elements are initialised to 0
  inline explicit Array(const IndexRange<num_dimensions>&);

  //! Construct an Array pointing to existing contiguous data
  /*!
    \arg data_sptr should point to a contiguous block of correct size.
    The constructed Array will essentially be a "view" of the
       \c data_sptr.get() block. Therefore, any modifications to the array will modify the data at \c data_sptr.get().
    This will be true until the Array is resized.

    The C-array \a data_ptr will be accessed with the last dimension running fastest
    ("row-major" order).
  */
  inline Array(const IndexRange<num_dimensions>& range, shared_ptr<elemT[]> data_sptr);

#ifndef SWIG
  // swig 2.0.4 gets confused by base_type (due to numeric template arguments)
  // therefore, we declare this constructor using the "self" type,
  // i.e. it's just a copy-constructor.
  // This is less powerful as in C++, but swig-generated interfaces don't need to know about the base_type anyway
  //! Construct an Array from an object of its base_type
  inline Array(const base_type& t);
#endif

  //! Copy constructor
  // implementation needed as the above doesn't disable the auto-generated copy-constructor
  inline Array(const self& t);

  //! virtual destructor, frees up any allocated memory
  inline ~Array() override;

  //! Swap content/members of 2 objects
  // implementation in .h because of templates/friends/whatever, see https://stackoverflow.com/a/61020224
  friend inline void swap(Array& first, Array& second) // nothrow
  {
    using std::swap;
    //  swap the members of two objects
    swap(static_cast<base_type&>(first), static_cast<base_type&>(second));
    swap(first._allocated_full_data_ptr, second._allocated_full_data_ptr);
  }

  //! move constructor
  /*! implementation uses the copy-and-swap idiom, see e.g. https://stackoverflow.com/a/3279550 */
  Array(Array&& other) noexcept;

  //! assignment operator
  /*! implementation uses the copy-and-swap idiom, see e.g. https://stackoverflow.com/a/3279550 */
  Array& operator=(Array other);

  /*! @name functions returning full_iterators*/
  //@{
  //! start value for iterating through all elements in the array, see full_iterator
  inline full_iterator begin_all();
  //! start value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator begin_all() const;
  //! start value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator begin_all_const() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline full_iterator end_all();
  //! end value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator end_all() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator end_all_const() const;
  //@}

  inline IndexRange<num_dimensions> get_index_range() const;

  //! return the total number of elements in this array
  inline size_t size_all() const;

  //! change the array to a new range of indices, new elements are set to 0
  /*! Current behaviour is that when resizing to a smaller array, the same memory
    will be used. However, when growing any of the dimensions, a new Array
    will be allocated and the data copied.

    If the array points to an existing block of data, resizing is therefore problematic.
    When growing the array, the resized array will no longer point to the original block
    of data.
  */
  inline virtual void resize(const IndexRange<num_dimensions>& range);

  //! alias for resize()
  virtual inline void grow(const IndexRange<num_dimensions>& range);

  //! return sum of all elements
  inline elemT sum() const;

  //! return sum of all positive elements
  inline elemT sum_positive() const;

  //! return maximum of all the elements
  inline elemT find_max() const;

  //! return minimum of all the elements
  inline elemT find_min() const;

  //! Fill elements with value \c n
  /*!
    hides VectorWithOffset::fill
   */
  inline void fill(const elemT& n);
  //! Sets elements below value to the value
  inline void apply_lower_threshold(const elemT& l);

  //! Sets elements above value to the value
  inline void apply_upper_threshold(const elemT& u);

  //! checks if the index range is 'regular'
  /*! Implementation note: this works by calling get_index_range().is_regular().
      We cannot rely on remembering if it was a regular range at construction (or
      even resizing) time as resize() could have been called on an element of the
      array. Checking for this would involve a performance penalty on operator[],
      which is definitely not a good idea.
  */
  inline bool is_regular() const;

  //! find regular range, returns \c false if the range is not regular
  /*! \see class IndexRange for a definition of (ir)regular ranges */
  bool get_regular_range(BasicCoordinate<num_dimensions, int>& min, BasicCoordinate<num_dimensions, int>& max) const;

  //! allow array-style access, read/write
  inline Array<num_dimensions - 1, elemT>& operator[](int i);

  //! array access, read-only
  inline const Array<num_dimensions - 1, elemT>& operator[](int i) const;

  //! allow array-style access given a BasicCoordinate to specify the indices, read/write
  inline elemT& operator[](const BasicCoordinate<num_dimensions, int>& c);

  //! array access given a BasicCoordinate to specify the indices, read-only
  // TODO alternative return value: elemT
  inline const elemT& operator[](const BasicCoordinate<num_dimensions, int>& c) const;

  //! \name indexed access with range checking (throw std:out_of_range)
  //@{
  inline Array<num_dimensions - 1, elemT>& at(int i);

  inline const Array<num_dimensions - 1, elemT>& at(int i) const;

  inline elemT& at(const BasicCoordinate<num_dimensions, int>& c);

  inline const elemT& at(const BasicCoordinate<num_dimensions, int>& c) const;
  //@}

  //! \deprecated a*x+b*y (use xapyb)
  template <typename elemT2>
  STIR_DEPRECATED inline void axpby(const elemT2 a, const Array& x, const elemT2 b, const Array& y);

  //! set values of the array to x*a+y*b, where a and b are scalar
  inline void xapyb(const Array& x, const elemT a, const Array& y, const elemT b);

  //! set values of the array to x*a+y*b, where a and b are arrays
  inline void xapyb(const Array& x, const Array& a, const Array& y, const Array& b);

  //! set values of the array to self*a+y*b where a and b are scalar or arrays
  template <class T>
  inline void sapyb(const T& a, const Array& y, const T& b);

  //! \name access to the data via a pointer
  //@{
  //! return if the array is contiguous in memory
  bool is_contiguous() const;

  //! member function for access to the data via a elemT*
  inline elemT* get_full_data_ptr();

  //! member function for access to the data via a const elemT*
  inline const elemT* get_const_full_data_ptr() const;

  //! signal end of access to elemT*
  inline void release_full_data_ptr();

  //! signal end of access to const elemT*
  inline void release_const_full_data_ptr() const;
  //@}

private:
  //! boolean to test if get_full_data_ptr is called
  // This variable is declared mutable such that get_const_full_data_ptr() can change it.
  mutable bool _full_pointer_access;

  //! A pointer to the allocated chunk if the array is constructed that way, zero otherwise
  shared_ptr<elemT[]> _allocated_full_data_ptr;

  //! change the array to a new range of indices, pointing to \c data_ptr
  /*!
    \arg data_ptr should point to a contiguous block of correct size

    The C-array \data_ptr will be accessed with the last dimension running fastest
    ("row-major" order).
  */
  inline void init(const IndexRange<num_dimensions>& range, elemT* const data_ptr, bool copy_data);
  // Make sure that we can access init() recursively
  template <int num_dimensions2, class elemT2>
  friend class Array;

  using base_type::grow;
  using base_type::resize;
};

/**************************************************
 (partial) specialisation for 1 dimensional arrays
 **************************************************/

//! The 1-dimensional (partial) specialisation of Array.
template <class elemT>
class Array<1, elemT> : public NumericVectorWithOffset<elemT, elemT>
#ifdef STIR_USE_BOOST
    ,
                        boost::operators<Array<1, elemT>, NumericVectorWithOffset<elemT, elemT>>,
                        boost::operators<Array<1, elemT>>,
                        boost::operators<Array<1, elemT>, elemT>
#endif
{
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
private:
#endif
  typedef NumericVectorWithOffset<elemT, elemT> base_type;
  typedef Array<1, elemT> self;

public:
  //! typedefs such that we do not need to have \c typename wherever we use these types defined in the base class
  //@{
  typedef typename base_type::value_type value_type;
  typedef typename base_type::reference reference;
  typedef typename base_type::const_reference const_reference;
  typedef typename base_type::difference_type difference_type;
  typedef typename base_type::size_type size_type;
  typedef typename base_type::iterator iterator;
  typedef typename base_type::const_iterator const_iterator;
  //@}

  /*! \brief Iterator type for going through all elements

    for 1D arrays, full iterators are equal to normal iterators
  */
  typedef iterator full_iterator;

  //! Iterator type for going through all elements of a const object
  typedef const_iterator const_full_iterator;

public:
  //! default constructor: array of length 0
  inline Array();

  //! constructor given an IndexRange<1>, initialising elements to 0
  inline explicit Array(const IndexRange<1>& range);

  //! constructor given first and last indices, initialising elements to 0
  inline Array(const int min_index, const int max_index);

  //! constructor given an IndexRange<1>, pointing to existing contiguous data
  /*!
    \arg data_ptr should point to a contiguous block of correct size.
    The constructed Array will essentially be a "view" of the
    \c data_sptr block. Therefore, any modifications to the array will modify the data at \a data_sptr.
    This will be the case until the Array is resized.
  */
  inline Array(const IndexRange<1>& range, shared_ptr<elemT[]> data_sptr);

  //! constructor given an IndexRange<1> from existing contiguous data (will copy)
  /*!
    \arg data_ptr should point to a contiguous block of correct size.
  */
  inline Array(const IndexRange<1>& range, const elemT* const data_ptr);

  //! constructor from basetype
  inline Array(const NumericVectorWithOffset<elemT, elemT>& il);

  //! Copy constructor
  // implementation needed as the above doesn't replace the normal copy-constructor
  // and the auto-generated is disabled because of the move constructor
  inline Array(const self& t);

  //! move constructor
  /*! implementation uses the copy-and-swap idiom, see e.g. https://stackoverflow.com/a/3279550 */
  Array(Array&& other) noexcept;

  //! virtual destructor
  inline ~Array() override;

  //! Swap content/members of 2 objects
  // implementation in .h because of templates/friends/whatever, see https://stackoverflow.com/a/61020224
  friend inline void swap(Array& first, Array& second) // nothrow
  {
    swap(static_cast<base_type&>(first), static_cast<base_type&>(second));
  }

  //! assignment
  inline Array& operator=(const Array& other);

  /*! @name functions returning full_iterators*/
  //@{
  //! start value for iterating through all elements in the array, see full_iterator
  inline full_iterator begin_all();
  //! start value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator begin_all() const;
  //! start value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator begin_all_const() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline full_iterator end_all();
  //! end value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator end_all() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator end_all_const() const;
  //@}

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! return the total number of elements in this array
  inline size_t size_all() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);

  // Array::grow initialises new elements to 0
  inline void grow(const int min_index, const int max_index) override;

  //! Array::resize initialises new elements to 0
  inline virtual void resize(const IndexRange<1>& range);

  inline void resize(const int min_index, const int max_index, bool initialise_with_0);
  //! resize, initialising new elements to 0
  inline void resize(const int min_index, const int max_index) override;

  //! \name access to the data via a pointer
  //@{
  //! return if the array is contiguous in memory (always \c true)
  bool is_contiguous() const
  {
    return true;
  }
  //! member function for access to the data via a elemT*
  inline elemT* get_full_data_ptr()
  {
    return this->get_data_ptr();
  }

  //! member function for access to the data via a const elemT*
  inline const elemT* get_const_full_data_ptr() const
  {
    return this->get_const_data_ptr();
  }

  //! signal end of access to elemT*
  inline void release_full_data_ptr()
  {
    this->release_data_ptr();
  }

  //! signal end of access to const elemT*
  inline void release_const_full_data_ptr() const
  {
    this->release_const_data_ptr();
  }
  //@}

  //! return sum of all elements
  inline elemT sum() const;

  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;

  //! return maximum value of all elements
  inline elemT find_max() const;

  //! return minimum value of all elements
  inline elemT find_min() const;

  //! checks if the index range is 'regular' (always \c true as this is the 1D case)
  inline bool is_regular() const;

  //! find regular range, returns \c false if the range is not regular
  bool get_regular_range(BasicCoordinate<1, int>& min, BasicCoordinate<1, int>& max) const;

#ifndef STIR_USE_BOOST

  /* KT 31/01/2000 I had to add these functions here, although they are
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be an Array, not
  its base_type (which happens if these function are not repeated
  in this class).
  Complicated...
  */
  //! elem by elem addition
  inline self operator+(const base_type& iv) const;

  //! elem by elem subtraction
  inline self operator-(const base_type& iv) const;

  //! elem by elem multiplication
  inline self operator*(const base_type& iv) const;

  //! elem by elem division
  inline self operator/(const base_type& iv) const;

  //! addition with an 'elemT'
  inline self operator+(const elemT a) const;

  //! subtraction with an 'elemT'
  inline self operator-(const elemT a) const;

  //! multiplication with an 'elemT'
  inline self operator*(const elemT a) const;

  //! division with an 'elemT'
  inline self operator/(const elemT a) const;

#endif // boost

  //! allow array-style access, read/write
  inline elemT& operator[](int i);

  //! array access, read-only
  inline const elemT& operator[](int i) const;

  //! allow array-style access giving its BasicCoordinate, read/write
  inline const elemT& operator[](const BasicCoordinate<1, int>& c) const;

  //! array access giving its BasicCoordinate, read-only
  inline elemT& operator[](const BasicCoordinate<1, int>& c);

  //! \name indexed access with range checking (throw std:out_of_range)
  //@{
  inline elemT& at(int i);

  inline const elemT& at(int i) const;

  inline elemT& at(const BasicCoordinate<1, int>& c);

  inline const elemT& at(const BasicCoordinate<1, int>& c) const;
  //@}
private:
  // Make sure we can call init() recursively.
  template <int num_dimensions2, class elemT2>
  friend class Array;

  //! change vector with new index range and point to \c data_ptr
  /*!
    \arg data_ptr should start to a contiguous block of correct size
  */
  inline void init(const IndexRange<1>& range, elemT* const data_ptr, bool copy_data);
};

END_NAMESPACE_STIR

#ifdef ARRAY_FULL
#  ifndef ARRAY_FULL2
#    include "FullArrayIterator.h"
#  else
#    include "FullArrayIterator2.h"
#    include "FullArrayConstIterator.h"
#  endif
#endif

#include "stir/Array.inl"

#endif // __Array_H__
