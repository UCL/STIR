
// $Id$: $Date$

#ifndef __Array_H__
#define __Array_H__

/*!
  \file 
  \ingroup buildblock 
  \brief defines the Array class for multi-dimensional (numeric) arrays

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$
  \version $Revision$

  Not all compilers support the full iterators, so they are disabled by  
  default.
*/
#include "NumericVectorWithOffset.h"
#include "ByteOrder.h"
#include "NumericType.h"
#include "IndexRange.h"

#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using std::istream;
using std::ostream;
#endif


START_NAMESPACE_TOMO
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
  \ingroup buildblock
  \brief This class defines multi-dimensional (numeric) arrays.

  This class implements multi-dimensional arrays which can have
'irregular' ranges. See IndexRange for a description of the ranges. Normal
numeric operations are defined. In addition, two types of iterators are
defined, one which iterators through the outer index, an done which
iterates through all elements of the array.

Array inherits its numeric operators from NumericVectorWithOffset.
In particular this means that operator+= etc. potentially grow
the object. However, as grow() is a virtual function, Array::grow is
called, which initialises new elements first to 0.
*/

template <int num_dimensions, typename elemT>
class Array : public NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT>
{
private:
  typedef  Array<num_dimensions, elemT> self;

protected:
  typedef NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT> base_type;
 
public:
#ifdef ARRAY_FULL
  typedef elemT value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
#ifndef ARRAY_FULL2
  //! This defines an iterator type that iterates through all elements.
  typedef FullArrayIterator<base_type::iterator, typename Array<num_dimensions-1, elemT>::full_iterator, elemT, reference, pointer> full_iterator;
#ifdef ARRAY_CONST_IT
  //! As full_iterator, but for const objects.
  typedef FullArrayIterator<base_type::const_iterator, typename Array<num_dimensions-1, elemT>::const_full_iterator, elemT, const_reference, const_pointer> const_full_iterator;
#else
  typedef FullArrayIterator<base_type::iterator, typename Array<num_dimensions-1, elemT>::full_iterator, elemT, reference, pointer> const_full_iterator;
#endif
#else // ARRAY_FULL2
  typedef FullArrayIterator<num_dimensions, elemT, reference, pointer> full_iterator;
#ifdef ARRAY_CONST_IT
  typedef FullArrayConstIterator<num_dimensions, elemT, const_reference, const_pointer> const_full_iterator;
#endif
#endif
#endif
  
public:
  //! Construct an empty Array
  inline Array();

  //! Construct an Array of given range of indices, elements are initialised to 0
  inline explicit Array(const IndexRange<num_dimensions>&);
  
  //! Construct an Array from an object of its base_type
  inline Array(const base_type& t);
  
  //! virtual destructor, frees up any allocated memory
  inline virtual ~Array();

#ifdef ARRAY_FULL
  //! start value for iterating through all elements in the array, see full_iterator
  inline full_iterator begin_all();
#ifdef ARRAY_CONST_IT  
  //! start value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator begin_all() const;
#endif  
  //! end value for iterating through all elements in the array, see full_iterator
  inline full_iterator end_all();
#ifdef  ARRAY_CONST_IT
  //! end value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator end_all() const;
#endif
#endif
  
  //! return the range of indices used  
  inline IndexRange<num_dimensions> get_index_range() const;

  //! grow the array to a new range of indices, new elements are set to 0  
  inline virtual void 
    grow(const IndexRange<num_dimensions>& range);
  
  //! return sum of all elements
  inline elemT sum() const ;
  
  //! return sum of all positive elements
  inline elemT sum_positive() const ;
  
  //! return maximum of all the elements
  inline elemT find_max() const;
  
  //! return minimum of all the elements
  inline elemT find_min() const;
  
  //! Fill elements with value n (overrides VectorWithOffset::fill)
  inline void fill(const elemT &n);

  //! checks if the index range is 'regular'
  inline bool is_regular() const;

  //! find regular range, returns false if the range is not regular
  bool get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max) const;
  
  //! read data from stream
  inline void read_data(istream& s, const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream
  inline void write_data(ostream& s, const ByteOrder byte_order = ByteOrder::native) const;    
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;

private:
 
  //! variable storing info on regularity
  bool is_regular_range;
};



/**************************************************
 (partial) specialisation for 1 dimensional arrays
 **************************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

//! The 1-dimensional (partial) specialisation of Array. 
template <class elemT>
class Array<1, elemT> : public NumericVectorWithOffset<elemT, elemT>
#ifdef TOMO_USE_BOOST
                         ,
			 boost::operators<Array<1, elemT>, NumericVectorWithOffset<elemT, elemT> >,
			 boost::operators<Array<1, elemT> >,
			 boost::operators<Array<1, elemT>, elemT>
#endif
{
protected: 
  
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;


public:

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
  inline Array(const base_type &il);
  
  //! virtual destructor
  inline virtual ~Array();

  //! start value for iterating through all elements in the array
  inline full_iterator begin_all();

  //! start value for iterating through all elements in the (const) array
  inline const_full_iterator begin_all() const;

  //! end value for iterating through all elements in the array
  inline full_iterator end_all();

  //! end value for iterating through all elements in the (const) array
  inline const_full_iterator end_all() const;

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
  //! checks if the index range is 'regular' (always true)
  inline bool is_regular() const;

  //! find regular range
  bool get_regular_range(
     BasicCoordinate<1, int>& min,
     BasicCoordinate<1, int>& max) const;

#ifndef TOMO_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
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
  
  //!  read data from stream, only valid for 'simple' type elemT    
  void read_data(istream& s, 
    const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream, only valid for 'simple' type elemT    
  void write_data(ostream& s,
		  const ByteOrder byte_order = ByteOrder::native) const;
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
  
};


#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/* 
  If the compiler does not support partial template specialisation, 
  we resort to multiple definitions of the class, for specific
  types of elemT.
  This of course means that if you want to use Array<n,elemT> for 'elemT'
  anything else then the types used defined here, you'll have to add 
  similar repetitions yourself...
  Currently supported for float, int, short, unsigned short
  */

/************************** float *********************/

template<>
class Array<1, float> : public NumericVectorWithOffset<float, float>
#ifdef TOMO_USE_BOOST
                         ,
			 boost::operators<Array<1, float>, NumericVectorWithOffset<float, float> >,
			 boost::operators<Array<1, float> >,
			 boost::operators<Array<1, float>, float>
#endif
{
protected: 
  typedef float elemT;
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;

public:  
  //! for 1D arrays, full iterators are equal to normal iterators
  typedef iterator full_iterator;

  //! for 1D arrays, full iterators are equal to normal iterators
  typedef const_iterator const_full_iterator;
    
public:
  
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor from basetype
  inline Array(const base_type &il);
  
  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! virtual destructor
  inline virtual ~Array();

  //! start value for iterating through all elements in the array
  inline full_iterator begin_all();

  //! start value for iterating through all elements in the (const) array
  inline const_full_iterator begin_all() const;

  //! end value for iterating through all elements in the array
  inline full_iterator end_all();

  //! end value for iterating through all elements in the (const) array
  inline const_full_iterator end_all() const;

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
#ifndef TOMO_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
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
  
  //!  read data from stream, only valid for 'simple' type elemT    
  void read_data(istream& s, 
    const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream, only valid for 'simple' type elemT    
  void write_data(ostream& s,
		  const ByteOrder byte_order = ByteOrder::native) const;
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
   
};

/**************************** int *************************/

template<>
class Array<1, int> : public NumericVectorWithOffset<int, int>
#ifdef TOMO_USE_BOOST
                         ,
			 boost::operators<Array<1, int>, NumericVectorWithOffset<int, int> >,
			 boost::operators<Array<1, int> >,
			 boost::operators<Array<1, int>, int>
#endif
{
protected: 
  typedef int elemT;
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;

  public:  
//! for 1D arrays, full iterators are equal to normal iterators
  typedef iterator full_iterator;

  //! for 1D arrays, full iterators are equal to normal iterators
  typedef const_iterator const_full_iterator;

  
public:
  
 
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor from basetype
  inline Array(const base_type &il);
  
  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! virtual destructor
  inline virtual ~Array();

  //! start value for iterating through all elements in the array
  inline full_iterator begin_all();

  //! start value for iterating through all elements in the (const) array
  inline const_full_iterator begin_all() const;

  //! end value for iterating through all elements in the array
  inline full_iterator end_all();

  //! end value for iterating through all elements in the (const) array
  inline const_full_iterator end_all() const;

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
#ifndef TOMO_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
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
  
  //!  read data from stream, only valid for 'simple' type elemT    
  void read_data(istream& s, 
    const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream, only valid for 'simple' type elemT    
  void write_data(ostream& s,
		  const ByteOrder byte_order = ByteOrder::native) const;
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
   
};

/******************************* short *************************/

template<>
class Array<1, short> : public NumericVectorWithOffset<short, short>
#ifdef TOMO_USE_BOOST
                         ,
			 boost::operators<Array<1, short>, NumericVectorWithOffset<short, short> >,
			 boost::operators<Array<1, short> >,
			 boost::operators<Array<1, short>, short>
#endif
{
protected: 
  typedef short elemT;
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;

  public:  
  //! for 1D arrays, full iterators are equal to normal iterators
  typedef iterator full_iterator;

  //! for 1D arrays, full iterators are equal to normal iterators
  typedef const_iterator const_full_iterator;

  
public:
  
 
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor from basetype
  inline Array(const base_type &il);
  
  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! virtual destructor
  inline virtual ~Array();

  //! start value for iterating through all elements in the array
  inline full_iterator begin_all();

  //! start value for iterating through all elements in the (const) array
  inline const_full_iterator begin_all() const;

  //! end value for iterating through all elements in the array
  inline full_iterator end_all();

  //! end value for iterating through all elements in the (const) array
  inline const_full_iterator end_all() const;

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
#ifndef TOMO_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
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
  
  //!  read data from stream, only valid for 'simple' type elemT    
  void read_data(istream& s, 
    const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream, only valid for 'simple' type elemT    
  void write_data(ostream& s,
		  const ByteOrder byte_order = ByteOrder::native) const;
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
   
};

/***************************** unsigned short ******************/
template<>
class Array<1, unsigned short> : public NumericVectorWithOffset<unsigned short, unsigned short>
#ifdef TOMO_USE_BOOST
                         ,
			 boost::operators<Array<1, unsigned short>, NumericVectorWithOffset<unsigned short, unsigned short> >,
			 boost::operators<Array<1, unsigned short> >,
			 boost::operators<Array<1, unsigned short>, unsigned short>
#endif
{
protected: 
  typedef unsigned short elemT;
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;
  
public:  
  //! for 1D arrays, full iterators are equal to normal iterators
  typedef iterator full_iterator;

  //! for 1D arrays, full iterators are equal to normal iterators
  typedef const_iterator const_full_iterator;
  
public:
   
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor from basetype
  inline Array(const base_type &il);

  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! virtual destructor
  inline virtual ~Array();

  //! start value for iterating through all elements in the array
  inline full_iterator begin_all();

  //! start value for iterating through all elements in the (const) array
  inline const_full_iterator begin_all() const;

  //! end value for iterating through all elements in the array
  inline full_iterator end_all();

  //! end value for iterating through all elements in the (const) array
  inline const_full_iterator end_all() const;

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;
  
  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
  
#ifndef TOMO_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
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
  
  //!  read data from stream, only valid for 'simple' type elemT    
  void read_data(istream& s, 
    const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream, only valid for 'simple' type elemT    
  void write_data(ostream& s,
		  const ByteOrder byte_order = ByteOrder::native) const;
  
#ifdef MEMBER_TEMPLATES
  template <class elemT2, class scaleT>
  void 
    read_data(istream& s, NumericInfo<elemT2> info2, scaleT& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  template <class elemT2, class scaleT>
  void 
    write_data(ostream& s, NumericInfo<elemT2> info2, scaleT& scale,
               const ByteOrder byte_order = ByteOrder::native) const;
#endif
  
  //! read data of different type from stream
  void 
    read_data(istream& s, NumericType type, float& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, float& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
   
};

#undef elemT

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_TOMO

#ifdef ARRAY_FULL
#  ifndef ARRAY_FULL2
#  include "FullArrayIterator.h"
#  else
#    include "FullArrayIterator2.h"
#    ifdef ARRAY_CONST_IT
#       include "FullArrayConstIterator.h"
#    endif
#  endif
#endif

#include "Array.inl"


#endif // __Array_H__
