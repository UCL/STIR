//
// $Id$
//

#ifndef __Tomo_VectorWithOffset_H__
#define __Tomo_VectorWithOffset_H__

/*!
  \file 
  \ingroup buildblock  
  \brief defines the VectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/


#include "tomo/common.h"

#include <iterator>
#ifndef TOMO_NO_NAMESPACES
using std::size_t;
using std::ptrdiff_t;
using std::random_access_iterator_tag;
#endif

START_NAMESPACE_TOMO


/*! 
  \ingroup buildblock
  \brief A templated class for vectors, but with indices starting not from 0

  Elements are guaranteed to be stored contiguously. (Emergency) methods
  are provided for accessing the data via a \c T*.

  \warning This class does not satisfy full Container requirements.
  \warning Current implementation relies on shifting a \c T* outside the range
  of allocated data. This is not guaranteed to be valid by ANSI C++. It is fine
  however as long as the min_index is negative and such that abs(min_index) is
  smaller than max_index.
*/

template <class T>
class VectorWithOffset
{
public:
  // typedefs for iterator support
  
  typedef random_access_iterator_tag iterator_category;  
  typedef T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;

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
  
  //! Free memory and make object as if default-constructed
  // TODO rename to recycle
  inline void Recycle();

  //! change value of starting index
  inline void set_offset(const int min_first);
  
  //! grow the range of the tensor, new elements are set to \c T()
  inline virtual void grow(const int min_index, const int max_index);
  
  //! assignment operator
  inline VectorWithOffset & operator= (const VectorWithOffset &il) ;
  
  //! return number of elements in this vector
  inline int get_length() const;	
  
  //! get value of first valid index
  inline int get_min_index() const;

  //! get value of last valid index
  inline int get_max_index() const;

  //! allow array-style access, read/write
  inline T& operator[] (int i);

  //! array access, read-only
  inline const T& operator[] (int i) const;
  
  //! comparison
  inline bool operator== (const VectorWithOffset &iv) const;
  inline bool operator!= (const VectorWithOffset &iv) const;
  
  //! fill elements with value n
  inline void fill(const T &n);
  
  //! member function for access to the data via a T*
  inline T* get_data_ptr();

  //! member function for access to the data via a const T*
#ifndef TOMO_NO_MUTABLE
  inline const T * get_const_data_ptr() const;
#else
  inline const T * get_const_data_ptr();
#endif

  //! signal end of access to T*
  inline void release_data_ptr();

  // basic iterator support

  //! use to initialise an iterator to the first element of the vector
  inline iterator begin();
  //! use to initialise an iterator to the first element of the (const) vector
  inline const_iterator begin() const;
  //! iterator 'past' the last element of the vector
  inline iterator end();
  //! iterator 'past' the last element of the (const) vector
  inline const_iterator end() const;


protected:
  
  //! length of vector
  int length;	
  //! starting index
  int start;	

  //! array to hold elements indexed by start
  T *num;	

  //! Default member settings for all constructors
  inline void init();
  
  //! Called internally to see if all variables are consistent
  inline void check_state() const;

  //! boolean to test if get_data_ptr is called
  // This variable is declared mutable such that get_const_data_ptr() can change it.
#ifndef TOMO_NO_MUTABLE
  mutable
#endif
  bool pointer_access;

};

END_NAMESPACE_TOMO

#include "VectorWithOffset.inl"

#endif // __VectorWithOffset_H__
