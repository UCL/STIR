
// $Id$: $Date$

#ifndef __Array_H__
#define __Array_H__

#include "Tomography_common.h"


#include "ByteOrder.h"
#include "NumericType.h"
#include "NumericVectorWithOffset.h"
#include "IndexRange.h"

START_NAMESPACE_TOMO

template <int num_dimensions, typename elemT>
class Array : public NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT>
{
protected:
  typedef NumericVectorWithOffset<Array<num_dimensions-1, elemT>, elemT> base_type;
  
public:
  //! Construct an empty Array
  inline Array();

  //! Construct an Array of given range of indices
  inline explicit Array(const IndexRange<num_dimensions>&);
  
  //! Construct an Array from an object of its base_type
  inline Array(const base_type& t);
  
  //! virtual destructor
  inline virtual ~Array();

  //! return the range of indices used  
  inline IndexRange<num_dimensions> get_index_range() const;

  //! grow the array to a new range of indices  
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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
};



/**************************************************
 (partial) specialisation for 1 dimensional arrays
 **************************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

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
  
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! constructor from basetype
  inline Array(const base_type &il);
  
  //! virtual destructor
  inline virtual ~Array();

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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
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

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(int hfirst, int hlast);
  
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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
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

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(int hfirst, int hlast);
  
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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
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

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(int hfirst, int hlast);
  
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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
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

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;
  
  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(int hfirst, int hlast);
  
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
    read_data(istream& s, NumericType type, Real& scale,
              const ByteOrder byte_order = ByteOrder::native);
  
  //! write data to stream as different type 
  void 
    write_data(ostream& s, NumericType type, Real& scale,
	       const ByteOrder byte_order = ByteOrder::native) const;
   
};

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_TOMO


#include "Array.inl"


#endif // __Array_H__
