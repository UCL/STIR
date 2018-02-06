//
//
/*!

  \file
  \ingroup Array
  \brief Declaration of class stir::ArrayFilter1DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2010, Hammersmith Imanet Ltd
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

#ifndef __stir_ArrayFilter1DUsingConvolution_H__
#define __stir_ArrayFilter1DUsingConvolution_H__


#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
#include "stir/BoundaryConditions.h"

START_NAMESPACE_STIR

template <typename elemT> class VectorWithOffset;

/*!
  \ingroup Array
  \brief This class implements convolution of a 1D array with an 
  arbitrary (i.e. potentially non-symmetric) kernel.

  Convolution is non-periodic:

  \f[ out_i = \sum_j kernel_j in_{i-j} \f] 

  Note that for most kernels, the above convention means that the zero-
  index of the kernel corresponds to the peak in the kernel. 

  By default, zero boundary conditions are used, i.e. elements of the input array 
  that are outside its index range are considered to be 0.   

  Currently, "constant" boundary conditions are also implemented, i.e. elements of the input array 
  that are outside its index range are considered to the same as the nearest element in the array
  (i.e. first element on the "left" and last element on the "right").

  \par Example 1
  A straightforward low-pass filter, with a symmetric kernel
  \code
  VectorWithOffset<float> kernel(-1,1);
  kernel[-1] = kernel[1] = .25F; kernel[0] = 0.5F;
  ArrayFilter1DUsingConvolution<float> lowpass_filter(kernel);
  \endcode
  \par Example 2
  A filter which shifts the output 1 index to the right, i.e. \f$ out_i = in_{i-1}\f$
  \code
  VectorWithOffset<float> kernel(1,1);
  kernel[1] = 1.F;
  ArrayFilter1DUsingConvolution<float> right_shift_filter(kernel);
  \endcode

  \warning 1 argument operator() currently leaves the array with the
  the same index range, i.e. it does not extend it with the kernel size or so.

  \see ArrayFilter1DUsingSymmetricConvolution for an implementation
  when the kernel is symmetric. (Note: it's not clear if that implementation
  results in faster execution).

  \todo implement other boundary conditions
  */
template <typename elemT>
class ArrayFilter1DUsingConvolution : 
  public ArrayFunctionObject_2ArgumentImplementation<1,elemT>
{
public:

  //! Construct a trivial filter
  ArrayFilter1DUsingConvolution();

  //! Construct the filter given the kernel coefficients
  /*! Currently \a bc has to be BoundaryConditions::zero or 
      BoundaryConditions::constant
  */
  ArrayFilter1DUsingConvolution(const VectorWithOffset< elemT>& filter_kernel, const BoundaryConditions::BC bc= BoundaryConditions::zero);
  //! checks if the kernel corresponds to a trivial filter operation
  /*! 
    trivial means, either the kernel has 0 length, or length 1 and its only element is 1
    */
  bool is_trivial() const;

  virtual Succeeded 
    get_influencing_indices(IndexRange<1>& influencing_indices, 
                            const IndexRange<1>& output_indices) const;

  virtual Succeeded 
    get_influenced_indices(IndexRange<1>& influenced_indices, 
                           const IndexRange<1>& input_indices) const;

private:
  VectorWithOffset< elemT> filter_coefficients;
  BoundaryConditions::BC _bc;
  void do_it(Array<1,elemT>& out_array, const Array<1,elemT>& in_array) const;

};



END_NAMESPACE_STIR


#endif //ArrayFilter1DUsingConvolution


