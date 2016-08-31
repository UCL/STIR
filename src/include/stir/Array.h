
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-10-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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

#ifndef __stir_Array_H__
#define __stir_Array_H__

#ifndef ARRAY_FULL
#define ARRAY_FULL
#endif

/*!
  \file 
  \ingroup Array 
  \brief defines the Array class for multi-dimensional (numeric) arrays

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project


  Not all compilers support the full iterators, so you could disabled them by editing 
  the file and removing the define ARRAY_FULL. Lots of other things in the library 
  won't work then.
*/
#include "stir/NumericVectorWithOffset.h"
#include "stir/ByteOrder.h"
#include "stir/IndexRange.h"   

START_NAMESPACE_STIR
class NumericType;

#ifdef ARRAY_FULL
#ifndef ARRAY_FULL2
template <typename topleveliterT, typename restiterT, typename elemT, typename _Ref, typename _Ptr>
class FullArrayIterator;
#else
template <int num_dimensions, typename elemT, typename _Ref, typename _Ptr>
class FullArrayIterator;
template <int num_dimensions, typename elemT, typename _Ref, typename _Ptr>
class FullArrayConstIterator;
#endif

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
class Array : public NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT>
{
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
 private: 
#endif  
  typedef  Array<num_dimensions, elemT> self;
  typedef NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT> base_type;
 
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
#ifndef ARRAY_FULL2
  //! This defines an iterator type that iterates through all elements.
  typedef FullArrayIterator<typename base_type::iterator, typename Array<num_dimensions-1, elemT>::full_iterator, elemT, full_reference, full_pointer> full_iterator;

  //! As full_iterator, but for const objects.
  typedef FullArrayIterator<typename base_type::const_iterator, typename Array<num_dimensions-1, elemT>::const_full_iterator, elemT, const_full_reference, const_full_pointer> const_full_iterator;
#else // ARRAY_FULL2
  typedef FullArrayIterator<num_dimensions, elemT, full_reference, full_pointer> full_iterator;

  typedef FullArrayConstIterator<num_dimensions, elemT, const_full_reference, const_full_pointer> const_full_iterator;

#endif
  //@} 
#endif
public:
  //! Construct an empty Array
  inline Array();

  //! Construct an Array of given range of indices, elements are initialised to 0
  inline explicit Array(const IndexRange<num_dimensions>&);
  
#ifndef SWIG
  //! Construct an Array from an object of its base_type
  inline Array(const base_type& t);
#else
  // swig 2.0.4 gets confused by base_type (due to numeric template arguments)
  // therefore, we declare this constructor using the "self" type, 
  // i.e. it's just a copy-constructor.
  // This is less powerful as in C++, but swig-generated interfaces don't need to know about the base_type anyway
  inline Array(const self& t);
#endif
  
  //! virtual destructor, frees up any allocated memory
  inline virtual ~Array();

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

  /* Implementation note: grow() and resize() are inline such that they are
     defined for any type you happen to use for elemT. Otherwise, we would
     need instantiation in Array.cxx.
  */
  //! change the array to a new range of indices, new elements are set to 0  
  inline virtual void 
    resize(const IndexRange<num_dimensions>& range);

  //! grow the array to a new range of indices, new elements are set to 0  
  virtual inline void 
    grow(const IndexRange<num_dimensions>& range);
  
  //! return sum of all elements
  inline elemT sum() const ;

  //! return the mean value of all elements
  inline float mean() const;
  
  //! return sum of all positive elements
  inline elemT sum_positive() const ;
  
  //! return maximum of all the elements
  inline elemT find_max() const;
                                                            
  //! return minimum of all the elements
  inline elemT find_min() const;
  
  //! Fill elements with value n (overrides VectorWithOffset::fill)
  inline void fill(const elemT &n);

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
  bool get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max) const;
   
  //! allow array-style access, read/write
  inline Array<num_dimensions-1, elemT>& 
	operator[] (int i);

  //! array access, read-only
  inline const Array<num_dimensions-1, elemT>& 
	operator[] (int i) const;

  //! allow array-style access given a BasicCoordinate to specify the indices, read/write  
  inline elemT&
  operator[](const BasicCoordinate<num_dimensions,int> &c) ;

  //! array access given a BasicCoordinate to specify the indices, read-only
  // TODO alternative return value: elemT
  inline const elemT&
  operator[](const BasicCoordinate<num_dimensions,int> &c) const;

  //! \name indexed access with range checking (throw std:out_of_range)
  //@{
  inline Array<num_dimensions-1, elemT>& 
    at (int i);

  inline const Array<num_dimensions-1, elemT>& 
    at (int i) const;

  inline elemT&
    at(const BasicCoordinate<num_dimensions,int> &c) ;

  inline const elemT&
    at(const BasicCoordinate<num_dimensions,int> &c) const;
  //@}
};



/**************************************************
 (partial) specialisation for 1 dimensional arrays
 **************************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

//! The 1-dimensional (partial) specialisation of Array. 
template <class elemT>
class Array<1, elemT> : public NumericVectorWithOffset<elemT, elemT>
#ifdef STIR_USE_BOOST
      ,
			 boost::operators<Array<1, elemT>, NumericVectorWithOffset<elemT, elemT> >,
			 boost::operators<Array<1, elemT> >,
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
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
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

  //! constructor from basetype
  inline Array(const NumericVectorWithOffset<elemT,elemT> &il);
  
  //! virtual destructor
  inline virtual ~Array();

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
  inline virtual void grow(const int min_index, const int max_index);
  
  //! Array::resize initialises new elements to 0
  inline virtual void resize(const IndexRange<1>& range);
  
  // Array::resize initialises new elements to 0
  inline virtual void resize(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;

  //! return the mean value of all elements
  inline float mean() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
  //! checks if the index range is 'regular' (always \c true as this is the 1D case)
  inline bool is_regular() const;
  
  //! find regular range, returns \c false if the range is not regular
  bool get_regular_range(
     BasicCoordinate<1, int>& min,
     BasicCoordinate<1, int>& max) const;
  
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
  inline self operator+ (const base_type &iv) const;
  
  //! elem by elem subtraction
  inline self operator- (const base_type &iv) const;
  
  //! elem by elem multiplication
  inline self operator* (const base_type &iv) const;
  
  //! elem by elem division
  inline self operator/ (const base_type &iv) const;
  
  //! addition with an 'elemT'
  inline self operator+ (const elemT a) const;
  
  //! subtraction with an 'elemT'
  inline self operator- (const elemT a) const;
  
  //! multiplication with an 'elemT'
  inline self operator* (const elemT a) const;
  
  //! division with an 'elemT'
  inline self operator/ (const elemT a) const;

#endif // boost

    //! allow array-style access, read/write
  inline elemT&	operator[] (int i);

  //! array access, read-only
  inline const elemT&	operator[] (int i) const;
    
  //! allow array-style access giving its BasicCoordinate, read/write  
  inline const elemT& operator[](const BasicCoordinate<1,int>& c) const;

  //! array access giving its BasicCoordinate, read-only
  inline elemT& operator[](const BasicCoordinate<1,int>& c) ;    

  //! \name indexed access with range checking (throw std:out_of_range)
  //@{
  inline elemT& 
    at (int i);

  inline const elemT& 
    at (int i) const;

  inline elemT&
    at(const BasicCoordinate<1,int> &c) ;

  inline const elemT&
    at(const BasicCoordinate<1,int> &c) const;
  //@}

  
};


#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/* 
  If the compiler does not support partial template specialisation, 
  we resort to multiple definitions of the class, for specific
  types of elemT.
  This of course means that if you want to use Array<n,elemT> for 'elemT'
  anything else then the types used defined here, you'll have to add 
  similar repetitions yourself...
  Currently supported for signed char, float, int, short, unsigned short
  */
#define elemT signed char
#include "stir/Array1d.h"
#define elemT short
#include "stir/Array1d.h"
#define elemT unsigned short
#include "stir/Array1d.h"
#define elemT int
#include "stir/Array1d.h"
#define elemT float
#include "stir/Array1d.h"
END_NAMESPACE_STIR
#include <complex>
START_NAMESPACE_STIR
#define __stir_Array1d_no_comparisons__
#define elemT std::complex<float>
#include "stir/Array1d.h"
#undef elemT
#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_STIR

#ifdef ARRAY_FULL
#  ifndef ARRAY_FULL2
#  include "FullArrayIterator.h"
#  else
#    include "FullArrayIterator2.h"
#       include "FullArrayConstIterator.h"
#  endif
#endif

#include "stir/Array.inl"


#endif // __Array_H__
