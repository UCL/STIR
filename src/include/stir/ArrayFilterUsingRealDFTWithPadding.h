//
// $Id$
//
/*!
  \file
  \ingroup Array
  \brief Declaration of class ArrayFilterUsingRealDFTWithPadding

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ArrayFilterUsingRealDFTWithPadding_H__
#define __stir_ArrayFilterUsingRealDFTWithPadding_H__


#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
#include "stir/IndexRange.h"
#include <complex>

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
  //! Construct the filter given the complex kernel coefficients
  /*! \see set_kernel(const Array<num_dimensions, std::complex<elemT> >&)
    \warning Will call error() when sizes are not appropriate
  */
  ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, std::complex<elemT> >& complex_filter_kernel);

  //! set the real kernel coefficients
  /*
    The kernel can be given with arbitrary index range, but will be wrapped-around, 
      assuming that it is periodic outside the indexrange of the kernel. So,
      normally, the 0- index corresponds to the middle of the PSF.

      Input data will be zero-padded to the same length as this kernel before
      DFT. If you want to avoid aliasing, make sure that the kernel is at least
      twice as long as the input and output arrays.

      As this function uses fourier(), see there for restrictions on the possible
      kernel length, but at time of writing, it has to be a power of 2.
  */
  Succeeded 
    set_kernel(const Array<num_dimensions, elemT>& real_filter_kernel);
  //! set the complex kernel coefficients
  /*  The kernel has to be given with index ranges starting from 0.
      So, the 0- index corresponds to the DC component of the filter.

      Input data will be zero-padded to the same length as this kernel before
      DFT. If you want to avoid aliasing, make sure that the kernel is at least
      twice as long as the input and output arrays.

      As this function uses fourier(), see there for restrictions on the possible
      kernel length, but at time of writing, it has to be a power of 2.
  */
  Succeeded
    set_kernel(const Array<num_dimensions, std::complex<elemT> >& complex_filter_kernel);

  //! checks if the kernel corresponds to a trivial filter operation
  /*! 
    trivial means, either the kernel has 0 length, or length 1 and its only element is 1
    */
  bool is_trivial() const;

private:
  Array<num_dimensions, std::complex<elemT> > complex_filter_kernel;
  IndexRange<num_dimensions> padding_range;
  void do_it(Array<num_dimensions, elemT>& out_array, const Array<num_dimensions, elemT>& in_array) const;
  Succeeded set_padding_range();
};


END_NAMESPACE_STIR


#endif //ArrayFilterUsingRealDFTWithPadding


