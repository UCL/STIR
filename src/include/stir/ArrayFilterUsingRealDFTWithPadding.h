/*!
  \file
  \ingroup Array
  \brief Declaration of class stir::ArrayFilterUsingRealDFTWithPadding

  \author Kris Thielemans
*/
/*
    Copyright (C) 2004-2009, Hammersmith Imanet Ltd
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

#ifndef __stir_ArrayFilterUsingRealDFTWithPadding_H__
#define __stir_ArrayFilterUsingRealDFTWithPadding_H__

#include "stir/Array.h"
#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
#include "stir/Array_complex_numbers.h"
#include "stir/IndexRange.h"
#include <complex>

#if defined(_MSC_VER) && _MSC_VER<=1200
/* VC 6.0 cannot overload the real and complex constructors
   I just disable the complex version.
   */
#define __stir_ArrayFilterUsingRealDFTWithPadding_no_complex_kernel__

#endif

START_NAMESPACE_STIR
class Succeeded;
template <int num_dimensions, typename elemT> class Array;

/*!
  \ingroup Array
  \brief This class implements convolution of an array of real numbers with an 
  arbitrary (i.e. potentially non-symmetric) kernel using DFTs.

  Convolution is periodic.

  Elements of the input array that are outside its
  index range are considered to be 0.   

  If the input and output arrays are smaller than half the length of the kernel,
  then there is enough zero-padding such that aliasing cannot occur.
  In this case, this class gives the same results as 
  ArrayFilter1DUsingConvolution.
  */
template <int num_dimensions, typename elemT>
class ArrayFilterUsingRealDFTWithPadding : 
  public ArrayFunctionObject_2ArgumentImplementation<num_dimensions,elemT>
{
public:

  //! Default constructor (trivial kernel)
  ArrayFilterUsingRealDFTWithPadding();
  //! Construct the filter given the real kernel coefficients
  /*! \see set_kernel(const Array<num_dimensions, elemT>&)
    \warning Will call error() when sizes are not appropriate
  */      
  ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, elemT>& real_filter_kernel);

#ifndef __stir_ArrayFilterUsingRealDFTWithPadding_no_complex_kernel__
  //! Construct the filter given the complex kernel coefficients
  /*! \see set_kernel(const Array<num_dimensions, std::complex<elemT> >&)
    \warning Will call error() when sizes are not appropriate.
    \warning This function is disabled for VC 6.0 because of compiler limitations
  */
  ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, std::complex<elemT> >& kernel_in_frequency_space);
#endif

  //! set the real kernel coefficients
  /*
    The kernel can be given with arbitrary (but regular) index range, 
    but will be wrapped-around, 
      assuming that it is periodic outside the indexrange of the kernel. So,
      normally, the 0- index corresponds to the middle of the PSF.

      Input data will be zero-padded to the same range as this kernel before
      DFT. If you want to avoid aliasing, make sure that the kernel is at least
      twice as long as the input and output arrays.

      As this function uses fourier_for_real_data(), see there for restrictions 
      on the possible kernel length, but at time of writing, it has to be a power of 2.
  */
  Succeeded 
    set_kernel(const Array<num_dimensions, elemT>& real_filter_kernel);

  //! set the complex kernel coefficients
  /*  The kernel has to be given with index ranges starting from 0.
      So, the 0- index corresponds to the DC component of the filter.

      \see fourier_for_real_data() for more info on the range of frequencies
      Input data will be zero-padded to the index range as the corresponding 
      'real' kernel before DFT. If you want to avoid aliasing, make sure that the kernel is at least
      twice as long as the input and output arrays.

      See fourier() for restrictions on the possible
      kernel length, but at time of writing, it has to be a power of 2.
  */
  Succeeded
    set_kernel_in_frequency_space(const Array<num_dimensions, std::complex<elemT> >& kernel_in_frequency_space);

  //! checks if the kernel corresponds to a trivial filter operation
  /*! 
    trivial means, either the kernel has 0 length, or length 1 and its only element is 1
    */
  bool is_trivial() const;

protected:

	Array<num_dimensions, std::complex<elemT> > kernel_in_frequency_space;

  //! Performs the convolution
  /*! \a in_array and \a out_array can have arbitrary (even non-regular)
      index ranges. However, they will copied (if necessary)
      using wrap-around to/from
      an array with the same dimensions as the 'real' kernel.
  */
  void do_it(Array<num_dimensions, elemT>& out_array, const Array<num_dimensions, elemT>& in_array) const;
private:
  IndexRange<num_dimensions> padding_range;
  BasicCoordinate<num_dimensions,int> padded_sizes;
  Succeeded set_padding_range();
};


END_NAMESPACE_STIR


#endif //ArrayFilterUsingRealDFTWithPadding


