//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::ArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_ArrayFunctionObject_H__
#define __stir_ArrayFunctionObject_H__


#include "stir/Succeeded.h"



START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class Array;
template <int num_dimensions> class IndexRange;
/*!
  \ingroup Array
  \brief A class for operations on n-dimensional Arrays
*/
template <int num_dimensions, typename elemT>
class ArrayFunctionObject
{
public:
  virtual ~ArrayFunctionObject() {}
  //! in-place modification
  /*! \warning Not all derived classes will be able to handle arbitrary index ranges
      for \a in_array.
   */
  virtual void operator() (Array<num_dimensions,elemT>& array) const = 0;
  //! result stored in another array 
  /*! \warning Not all derived classes will be able to handle arbitrary index ranges
      in \a out_array and \a in_array.
   */
  virtual void operator() (Array<num_dimensions,elemT>& out_array, 
                           const Array<num_dimensions,elemT>& in_array) const = 0;
  //! Should return true when the operations won't modify the object at all
  /*! For the 2 argument version, elements in \a out_array will be set to 
      corresponding elements in \a in_array. Elements in \a out_array that do not
      occur in \a in_array will be set to 0.
  */
  virtual bool is_trivial() const  = 0;

  //! sets the range of indices that influences the result in a set of coordinates \a output_indices
  /*! For linear filters, these are the indices such that the support of their PSF 
      overlaps with output_indices.
      \return Succeeded::yes if this is a meaningful concept for the current object.
              Presumably, Succeeded::no would be returned if the whole array is always
              going to affect the \a output_indices (independent of the size of the input array) or
              of it is too difficult for the derived class to return a sensible index range.
   */
  virtual Succeeded 
    get_influencing_indices(IndexRange<num_dimensions>& influencing_indices, 
                            const IndexRange<num_dimensions>& output_indices) const
  { return Succeeded::no; }

  //! sets the range of indices that gets influenced by a set of coordinate \a input_indices
  /*! For linear filters, this is the union of the supports of the PSF for all \a output_indices.
      \return Succeeded::yes if this is a meaningful concept for the current object.
              Presumably, Succeeded::no would be returned if the whole array is always
              going to be affected by the \a input_indices (independent of the size of the output array) or
              of it is too difficult for the derived class to return a sensible index range.
   */
  virtual Succeeded 
    get_influenced_indices(IndexRange<num_dimensions>& influenced_indices, 
                           const IndexRange<num_dimensions>& input_indices) const
  { return Succeeded::no; }
};




END_NAMESPACE_STIR 

#endif 
