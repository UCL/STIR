//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_numerics_MatrixFunction_H__
#define __stir_numerics_MatrixFunction_H__
/*!
  \file
  \ingroup numerics
  
  \brief Declaration of functions for matrices
    
  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR


//----------------------------------------------------------------------
/*! \ingroup numerics
  \name functions specific for 1D Arrays
*/
//@{

//! Inner product of 2 1D arrays
/*! \ingroup numerics 
  This returns the sum of multiplication of elements of \a conjugate(v1) and \a v2.

  Implementation is appropriate for complex numbers.

  Arguments must have the same index range.
 */
template<class elemT>
inline elemT 
inner_product (const Array<1,elemT> & v1, const Array<1,elemT> &v2);

//! angle between 2 1D arrays
/*! \ingroup numbers 
 */
template<class elemT>
inline double
angle (const Array<1,elemT> & v1, const Array<1,elemT> &v2);

//@} end of 1D functions


//----------------------------------------------------------------------
/*! \ingroup numerics
  \name functions for matrices
*/
//@{

//! matrix with vector multiplication
template <class elemT>	
inline Array<1,elemT> 
  matrix_multiply(const Array<2,elemT>& m, const Array<1,elemT>& vec);

//! matrix multiplication
template <class elemT>	
inline Array<2,elemT>
  matrix_multiply(const Array<2,elemT> &m1, const Array<2,elemT>& m2);

//! matrix transposition
template <class elemT>
inline Array<2,elemT>
  matrix_transpose (const Array<2,elemT>& m);

//! construct a diagonal matrix with all elements on the diagonal equal
/*! \param[in] dimension specifies the size of the matrix
   \param[in] value gives the value on the diagonal. Note that its 
    type determines the type of the return value.

    \par Example
    \code
    // a 3x3 identity matrix
    Array<2,float> iden = diagonal_matrix(3, 1.F);
    \endcode

    Index-range of the matrix will be <code>0</code> till 
    <code>dimensions-1</code>.
*/
template <class elemT>
inline 
Array<2,elemT>
  diagonal_matrix(const unsigned dimension, const elemT value);

//! construct a diagonal matrix 
/*! 
   \param[in] values gives the values on the diagonal. Note that its 
    type determines the type of the return value.

    \par Example
    \code
    // a 3x3 diagonal matrix with values 1,2,3 on the diagonal
    Array<2,float> iden = diagonal_matrix(Coordinate3D<float>(1,2,3));
    \endcode

    Index-range of the matrix will be <code>0</code> till 
    <code>dimensions-1</code>. (Note that this is different from
    \a values).
*/
template <int dimension, class elemT>
inline 
Array<2,elemT>
  diagonal_matrix(const BasicCoordinate<dimension,elemT>& values);

//@}

END_NAMESPACE_STIR

#include "stir/numerics/MatrixFunction.inl"

#endif
