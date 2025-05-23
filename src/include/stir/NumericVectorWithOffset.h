//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2005-06-03, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2020, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __NumericVectorWithOffset_H__
#define __NumericVectorWithOffset_H__
/*!
  \file 
 
  \brief defines the stir::NumericVectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project
  \author Gemma Fardell

*/


#include "stir/VectorWithOffset.h"
#include "stir/deprecated.h"

START_NAMESPACE_STIR
/*! 
  \ingroup Array
  \brief like VectorWithOffset, but with changes in various numeric operators 

  Compared to VectorWithOffset, the numeric operators +,-,*,/ with 
  NumericVectorWithOffset objects return an object with the largest
  index range of its arguments. Similarly operators +=,-=,*=,/=
  potentially grow the <code>*this</code> object.

  In addition, operators +=,-=,*=,/= with objects of type \c elemT are defined.

  \warning It is likely that the automatic growing feature will be removed at some point.
 */
 
template <class T, class elemT>
class NumericVectorWithOffset : public VectorWithOffset<T>
{
private:
  typedef VectorWithOffset<T> base_type;

public:
  //! Construct an empty NumericVectorWithOffset
  inline NumericVectorWithOffset();

  //! Construct a NumericVectorWithOffset of given length
  inline explicit NumericVectorWithOffset(const int hsz);
    
  //! Construct a NumericVectorWithOffset of elements with offset \c min_index
  inline NumericVectorWithOffset(const int min_index, const int max_index);

  //! Constructor from an object of this class' base_type
  inline NumericVectorWithOffset(const VectorWithOffset<T>& t);

  // arithmetic operations with a vector, combining element by element

  //! adding vectors, element by element
  inline NumericVectorWithOffset operator+ (const NumericVectorWithOffset &v) const;

  //! subtracting vectors, element by element
  inline NumericVectorWithOffset operator- (const NumericVectorWithOffset &v) const;

  //! multiplying vectors, element by element
  inline NumericVectorWithOffset operator* (const NumericVectorWithOffset &v) const;

  //! dividing vectors, element by element
  inline NumericVectorWithOffset operator/ (const NumericVectorWithOffset &v) const;


  // arithmetic operations with a elemT
  // TODO??? use member templates 

  //! return a new vector with elements equal to the sum of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator+ (const elemT &v) const;

  //! return a new vector with elements equal to the difference of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator- (const elemT &v) const;

  //! return a new vector with elements equal to the multiplication of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator* (const elemT &v) const;
	       
  //! return a new vector with elements equal to the division of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator/ (const elemT &v) const;


  // corresponding assignment operators

  //! adding elements of \c v to the current vector
  inline NumericVectorWithOffset & operator+= (const NumericVectorWithOffset &v);

  //! subtracting elements of \c v from the current vector
  inline NumericVectorWithOffset & operator-= (const NumericVectorWithOffset &v);

  //! multiplying elements of the current vector with elements of \c v 
  inline NumericVectorWithOffset & operator*= (const NumericVectorWithOffset &v);

  //! dividing all elements of the current vector by elements of \c v
  inline NumericVectorWithOffset & operator/= (const NumericVectorWithOffset &v);

  //! adding an \c elemT to the elements of the current vector
  inline NumericVectorWithOffset & operator+= (const elemT &v);

  //! subtracting an \c elemT from the elements of the current vector
  inline NumericVectorWithOffset & operator-= (const elemT &v);

  //! multiplying the elements of the current vector with an \c elemT 
  inline NumericVectorWithOffset & operator*= (const elemT &v);

  //! dividing the elements of the current vector by an \c elemT 
  inline NumericVectorWithOffset & operator/= (const elemT &v);

  //! \deprecated a*x+b*y (\see xapyb)
  template <typename elemT2>
    STIR_DEPRECATED inline void axpby(const elemT2 a, const NumericVectorWithOffset& x,
                      const elemT2 b, const NumericVectorWithOffset& y);
                      
  //! set values of the array to x*a+y*b, where a and b are scalar
  inline void xapyb(const NumericVectorWithOffset& x, const elemT a,
                    const NumericVectorWithOffset& y, const elemT b);

  //! set the values of the array to x*a+y*b, where a and b are vectors
  inline void xapyb(const NumericVectorWithOffset& x, const NumericVectorWithOffset& a,
                    const NumericVectorWithOffset& y, const NumericVectorWithOffset& b); 

  //! set the values of the array to self*a+y*b, where a and b are scalar or vectors
  template <class T2>
  inline void sapyb(const T2& a,
                    const NumericVectorWithOffset& y, const T2& b); 
};

END_NAMESPACE_STIR

#include "stir/NumericVectorWithOffset.inl"

#endif // __NumericVectorWithOffset_H__
