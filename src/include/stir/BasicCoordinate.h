#ifndef __stir_BasicCoordinate_H__
#define __stir_BasicCoordinate_H__
//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-01-04, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-01 - $Date$, Kris Thielemans
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

/*!
  \file 
  \ingroup Coordinate
 
  \brief This file declares class stir::BasicCoordinate and 
  some functions acting on stir::BasicCoordinate objects.

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$

  \todo The arithmetic operations might at some point be moved to a
  derived class stir::CartesianCoordinate. 

*/


#include "stir/common.h"
#include <boost/operators.hpp>
#include <iterator>

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
  //! \name typedefs for iterator support
  //@{
  typedef std::random_access_iterator_tag iterator_category;
  typedef coordT value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef value_type* pointer;
  typedef const value_type* const_iterator;
  typedef std::ptrdiff_t difference_type;
  typedef std::size_t size_type;
  //@}

  //! default constructor. NO initialisation
  inline BasicCoordinate();
  //! constructor that sets all elements equal to \a value.
  explicit inline BasicCoordinate(const coordT&);
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
	(*this)[i] = static_cast<coordT>(c[i]);
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
  //! Return value at index \c t (which is 1-based)
  inline coordT& operator[](const int d);
  //! Return value at index \c t (which is 1-based) if the BasicCoordinate object is const
  /*! Note that the return type is not simply \c coordT. This mimics the design of
      std::vector. One can argue about this (see e.g.
http://groups.google.com/group/comp.lang.c%2B%2B.moderated/browse_thread/thread/e5c4898a5c259cc1/434f5a25df51781f%23434f5a25df51781f?sa=X&oi=groupsr&start=2&num=3),
     However, this alternative can have severe performance penalties if \c coordT
     is a type for large objects.
  */
  inline coordT const& operator[](const int d) const;

  //! Return value at index \c t (which is 1-based), but with range checking (throws std::out_of_range)
  inline coordT& at(const int d);
  //! Return value at index \c t (which is 1-based) if the BasicCoordinate object is const, but with range checking (throws std::out_of_range)
  inline coordT const& at(const int d) const;

  // check if the coordinate is empty (always returns false)
  inline bool empty() const { return false; }

  //! \name Functions as in VectorWithOffset
  //@{
  inline static int get_min_index();
  inline static int get_max_index();
  inline static unsigned size();
  //! fill elements with value
  inline void fill(const coordT&);
  //@}

  //! \name arithmetic assignment operators
  //@{
  inline BasicCoordinate & operator+= (const BasicCoordinate& c);
  inline BasicCoordinate & operator-= (const BasicCoordinate& c);
  inline BasicCoordinate & operator*= (const BasicCoordinate& c);
  inline BasicCoordinate & operator/= (const BasicCoordinate& c);

  inline BasicCoordinate & operator+= (const coordT& a);
  inline BasicCoordinate & operator-= (const coordT& a);
  inline BasicCoordinate & operator*= (const coordT& a);
  inline BasicCoordinate & operator/= (const coordT& a);
  //@}

  //! \name arithmetic operations with a BasicCoordinate, combining element by element
  //@{
  inline BasicCoordinate operator+ (const BasicCoordinate& c) const;
  inline BasicCoordinate operator- (const BasicCoordinate& c) const;
  inline BasicCoordinate operator* (const BasicCoordinate& c) const;
  inline BasicCoordinate operator/ (const BasicCoordinate& c) const;
  //@}

  //! \name arithmetic operations with a coordT
  //@{
  inline BasicCoordinate operator+ (const coordT& a) const;
  inline BasicCoordinate operator- (const coordT& a) const;
  inline BasicCoordinate operator* (const coordT& a) const;	      
  inline BasicCoordinate operator/ (const coordT& a) const;
  //@}

  //! \name basic iterator support
  //@{
  inline iterator begin();
  inline const_iterator begin() const;
  inline iterator end();
  inline const_iterator end() const;
  //@}

 private:
  //! storage
  /*! \warning 0-based */
  coordT coords[num_dimensions];

};

/*!
  \ingroup Coordinate
  \name Utility functions to make BasicCoordinate objects.

  \warning Because of template rules of C++, all arguments of the
  make_coordinate function have to have exactly the same type. For instance,
  \code
  const BasicCoordinate<3,float> a = make_coordinate(1.F,2.F,0.F);
  \endcode
*/
//@{

template <class T>
inline BasicCoordinate<1,T>
  make_coordinate(const T& a1);

template <class T>
inline BasicCoordinate<2,T>
  make_coordinate(const T& a1, const T& a2);

template <class T>
inline BasicCoordinate<3,T>
  make_coordinate(const T& a1, const T& a2, const T& a3);

template <class T>
inline BasicCoordinate<4,T>
  make_coordinate(const T& a1, const T& a2, const T& a3, const T& a4);

template <class T>
inline BasicCoordinate<5,T>
  make_coordinate(const T& a1, const T& a2, const T& a3, const T& a4, const T& a5);

template <class T>
inline BasicCoordinate<6,T>
make_coordinate(const T& a1, const T& a2, const T& a3, const T& a4, const T& a5,
		const T& a6);

//@}

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

