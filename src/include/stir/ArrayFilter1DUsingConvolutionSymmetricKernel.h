//
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::ArrayFilter1DUsingConvolutionSymmetricKernel

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#ifndef __stir_ArrayFilter1DUsingConvolutionSymmetricKernel_H__
#define __stir_ArrayFilter1DUsingConvolutionSymmetricKernel_H__


#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"

START_NAMESPACE_STIR

template <typename elemT> class VectorWithOffset;

/*!
  \ingroup Array
  \brief This class implements convolution of a 1D array with a symmetric kernel.

  Convolution is non-periodic:

  \f[ out_i = \sum_j kernel_j in_{i+j} \f] 

  Elements of the input array that are outside its
  index range are considered to be 0.   

  \warning 2 argument operator() currently requires that out_array and in_array 
  have the same index range
  */
template <typename elemT>
class ArrayFilter1DUsingConvolutionSymmetricKernel : 
  public ArrayFunctionObject_2ArgumentImplementation<1,elemT>
{
public:

  //! Construct the filter given the kernel coefficients
  /*! 
    Only one half of the kernel coefficients has to be passed. The implementation
    uses a kernel whose coefficients are given by 
    \code kernel[i] == filter_kernel[abs(i)] \endcode

    \warning filter_kernel's indices must start from 0 
  */
  ArrayFilter1DUsingConvolutionSymmetricKernel(const VectorWithOffset< elemT>& filter_kernel);
  //! checks if the kernel corresponds to a trivial filter operation
  /*! 
    trivial means, either the kernel has 0 length, or length 1 and its only element is 1
    */
  bool is_trivial() const;

private:
  VectorWithOffset< elemT> filter_coefficients;
  void do_it(Array<1,elemT>& out_array, const Array<1,elemT>& in_array) const;

};



END_NAMESPACE_STIR


#endif //ArrayFilter1DUsingConvolutionSymmetricKernel


