//
//
/*
    Copyright (C) 2004- 2008, Hammersmith Imanet Ltd
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

#ifndef __stir_array_index_functions_h_
#define __stir_array_index_functions_h_

/*!
  \file 
  \ingroup Array
 
  \brief a variety of useful functions for indexing stir::Array objects

  \author Kris Thielemans

*/


#include "stir/Array.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

#if !defined(_MSC_VER) || _MSC_VER>1200
// VC 6.0 needs ugly work-arounds. We'll put these in the .inl file

/* \ingroup Array
   \name Functions for writing generic code with indexing of multi-dimensional arrays
   */
//@{ 

//! an alternative for array indexing using BasicCoordinate objects
/*! Case where the index has lower dimension than the array*/
template <int num_dimensions, int num_dimensions2, typename elemT>
inline
const Array<num_dimensions-num_dimensions2,elemT>&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions2,int> &c);

//! an alternative for array indexing using BasicCoordinate objects
/*! Case where the index has the same dimension as the array*/
template <int num_dimensions, typename elemT>
inline
const elemT&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions,int> &c);


//! Get the first multi-dimensional index of the array
/*! \todo If the array \arg a is empty, we return an object where all indices are 0.
    It would be better to throw an exception.
*/
template <int num_dimensions, typename T>
inline
BasicCoordinate<num_dimensions, int>
get_min_indices(const Array<num_dimensions, T>& a);


//! Given an index into an array, increment it to the next one
/*!
    \return \c true if the next index was still within the array, \c false otherwise

    This can be used to iterate through an array using code such as
    \code
    Array<num_dimensions2, T> array = ...;
    BasicCoordinate<num_dimensions, int> indices =
       get_min_indices(array);
    do 
    { something with indices }
    while (next(indices, array));
    \endcode
    \warning The above loop will fail for empty arrays
*/
template <int num_dimensions, typename T, int num_dimensions2>
inline
bool 
next(BasicCoordinate<num_dimensions, int>& indices, 
     const Array<num_dimensions2, T>& a);

//@}
#endif // end of VC 6.0 conditional


END_NAMESPACE_STIR

#include "stir/array_index_functions.inl"

#endif
