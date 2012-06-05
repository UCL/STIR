//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2005-06-03, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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

#ifndef __NumericVectorWithOffset_H__
#define __NumericVectorWithOffset_H__
/*!
  \file 
 
  \brief defines the stir::NumericVectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/


#include "stir/VectorWithOffset.h"

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

};

END_NAMESPACE_STIR

#include "stir/NumericVectorWithOffset.inl"

#endif // __NumericVectorWithOffset_H__
