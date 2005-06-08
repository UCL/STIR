#ifndef __stir_BasicCoordinate_H__
#define __stir_BasicCoordinate_H__
//
// $Id$
//

/*!
  \file 
  \ingroup Coordinate
 
  \brief This file declares class BasicCoordinate<num_dimensions, coordT> and 
  some functions acting on BasicCoordinate objects.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$

  \todo The arithmetic operations will at some point be moved to a
  derived class CartesianCoordinate. 

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/common.h"
#include <boost/operators.hpp>
#include <iterator>

#ifndef STIR_NO_NAMESPACES
using std::size_t;
using std::ptrdiff_t;
using std::random_access_iterator_tag;
#endif

START_NAMESPACE_STIR
/*!
  \ingroup Coordinate
  \brief class BasicCoordinate\<\c int \c num_dimensions, \c typename \c coordT\> defines \c num_dimensions -dimensional coordinates.


   A BasicCoordinate\<\c num_dimensions, \c coordT\> is essentially a vector of size \c num_dimensions, but 
   as the dimension is templated, it has better performance.

   Access to the individual coordinates is through operator[].
   
   \warning  Indices run from 1 to \c num_dimensions

*/
template <int num_dimensions, typename coordT>
  class BasicCoordinate :
  boost::partially_ordered<BasicCoordinate<num_dimensions, coordT>,  // have operator>, <= etc for free
  boost::equality_comparable<BasicCoordinate<num_dimensions, coordT> // have operator!= for free
  > >
{ 

public:
  // typedefs for iterator support
  typedef random_access_iterator_tag iterator_category;
  typedef coordT value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
	
  //! default constructor. NO initialisation
  inline BasicCoordinate();
#if 0
  /* disabled. It overlaps with the constructor below and VC 6.0 can't sort it out.
    We don't seem to need it anyway
  */
  //! copy constructor
  inline BasicCoordinate(const BasicCoordinate& c);
#endif
  //! constructor from different type of \c coordT
  /*! Uses assignment after <pre>static_cast<coordT></pre> */
  // Note implementation here for VC 6.0
  template <typename coordT2>
  explicit inline 
    BasicCoordinate(const BasicCoordinate<num_dimensions, coordT2>& c)
    {
      for (int i=1; i<=num_dimensions;++ i)
	coords[i] = static_cast<coordT>(c[i]);
    }
  // virtual destructor, not implemented at the moment 
  // It's only needed whenever there is a chance that an object will be deleted
  // through a pointer to its base class and derived classes have extra
  // data members
  // virtual ~BasicCoordinate() {}

  //! assignment
  inline BasicCoordinate & operator=(const BasicCoordinate& c);

  //! comparison
  inline bool operator==(const BasicCoordinate& c) const;
  
#if !defined(_MSC_VER) || _MSC_VER>1200
  //! less-than (using lexical ordering)
  inline bool operator<(const BasicCoordinate& c) const;
#else
  // needs to be a global function to have the overloading to work. sigh.
#endif

  // access to elements
  inline coordT& operator[](const int d);
  inline coordT operator[](const int d) const;

  // arithmetic assignment operators
  inline BasicCoordinate & operator+= (const BasicCoordinate& c);
  inline BasicCoordinate & operator-= (const BasicCoordinate& c);
  inline BasicCoordinate & operator*= (const BasicCoordinate& c);
  inline BasicCoordinate & operator/= (const BasicCoordinate& c);

  inline BasicCoordinate & operator+= (const coordT& a);
  inline BasicCoordinate & operator-= (const coordT& a);
  inline BasicCoordinate & operator*= (const coordT& a);
  inline BasicCoordinate & operator/= (const coordT& a);

  // arithmetic operations with a BasicCoordinate, combining element by element
  inline BasicCoordinate operator+ (const BasicCoordinate& c) const;
  inline BasicCoordinate operator- (const BasicCoordinate& c) const;
  inline BasicCoordinate operator* (const BasicCoordinate& c) const;
  inline BasicCoordinate operator/ (const BasicCoordinate& c) const;

  // arithmetic operations with a coordT
  inline BasicCoordinate operator+ (const coordT& a) const;
  inline BasicCoordinate operator- (const coordT& a) const;
  inline BasicCoordinate operator* (const coordT& a) const;	      
  inline BasicCoordinate operator/ (const coordT& a) const;

  // basic iterator support
  inline iterator begin();
  inline const_iterator begin() const;
  inline iterator end();
  inline const_iterator end() const;

 private:
  // allocate 1 too many to leave space for coords[0] (which is never used)
  coordT coords[num_dimensions+1];

};


/*
  General functions on BasicCoordinate objects, like in ArrayFunction.h
*/

//! compute  sum_i p1[i] * p2[i]
/*! 
  \ingroup Coordinate
  \todo move to a new CartesianCoordinate class */
template <int num_dimensions, typename coordT>
inline coordT
inner_product (const BasicCoordinate<num_dimensions, coordT>& p1, 
	       const BasicCoordinate<num_dimensions, coordT>& p2);
//! compute (inner_product(p1,p1))
/*! 
  \ingroup Coordinate
  \todo move to a new CartesianCoordinate class */
template <int num_dimensions, typename coordT>
inline double
norm_squared (const BasicCoordinate<num_dimensions, coordT>& p1);
//! compute sqrt(inner_product(p1,p1))
/*! 
  \ingroup Coordinate
  \todo move to a new CartesianCoordinate class */
template <int num_dimensions, typename coordT>
inline double
norm (const BasicCoordinate<num_dimensions, coordT>& p1);
//! compute angle between 2 directions
/*! \ingroup Coordinate
	Implemented in terms of acos(cos_angle(p1,p2)).
    \todo move to a new CartesianCoordinate class */
template <int num_dimensions, typename coordT>
inline double 
angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
       const BasicCoordinate<num_dimensions, coordT>& p2);

//! compute cos of the angle between 2 directions
/*! \ingroup Coordinate
    \todo move to a new CartesianCoordinate class      
*/
template <int num_dimensions, typename coordT>
inline double 
cos_angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
          const BasicCoordinate<num_dimensions, coordT>& p2);

// Note: gcc 2.8.1 bug:
// It cannot call 'join' (it generates a bad mangled name for the function)
//! make a longer BasicCoordinate, by prepending \c  c with the single \c coordT
/*! \ingroup Coordinate */
template <int num_dimensions, typename coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
join(const coordT& a, 
     const BasicCoordinate<num_dimensions, coordT>& c);    

//! make a longer BasicCoordinate, by appending the \c coordT to  \c c 
/*! \ingroup Coordinate */
template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
  join(const BasicCoordinate<num_dimensions, coordT>& c, const coordT& a);

//! make a shorter BasicCoordinate, by cutting the last element from \c c  
/*! \ingroup Coordinate */
template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions-1, coordT> 
  cut_last_dimension(const BasicCoordinate<num_dimensions, coordT>& c);


//! make a shorter BasicCoordinate, by cutting the first element from \c c  
/*! \ingroup Coordinate */
template <int num_dimensions, typename coordT>
inline BasicCoordinate<num_dimensions-1, coordT> 
cut_first_dimension(const BasicCoordinate<num_dimensions, coordT>& c) ;    

//! converts a BasicCoordinate<int> to BasicCoordinate<float>
/*! \ingroup Coordinate */
template<int num_dimensions>
inline 
BasicCoordinate<num_dimensions,float> 
convert_int_to_float(const BasicCoordinate<num_dimensions,int>& cint);


END_NAMESPACE_STIR

#include "stir/BasicCoordinate.inl"



#endif

