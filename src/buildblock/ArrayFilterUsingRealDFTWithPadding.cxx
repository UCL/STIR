//
// $Id$
//
/*!
  \file
  \ingroup Array
  \brief Implementation of class ArrayFilterUsingRealDFTWithPadding

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/Array.h"
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/numerics/fourier.h"
#include "stir/Succeeded.h"
#include "stir/BasicCoordinate.h"
#include "stir/array_index_functions.h"
#include "stir/modulo.h"
#include <algorithm>


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
ArrayFilterUsingRealDFTWithPadding()
{}

template <int num_dimensions, typename elemT>
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, elemT>& real_filter_kernel)
{ 
  if (set_kernel(real_filter_kernel) == Succeeded::no) 
    error("Error constructing ArrayFilterUsingRealDFTWithPadding\n");
}

#ifndef __stir_ArrayFilterUsingRealDFTWithPadding_no_complex_kernel__
template <int num_dimensions, typename elemT>
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, 
				   std::complex<elemT> >& complex_filter_kernel)
{ 
  if (set_kernel(complex_filter_kernel) == Succeeded::no) 
    error("Error constructing ArrayFilterUsingRealDFTWithPadding\n");
}
#endif

template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_padding_range()
{
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;

  if (!complex_filter_kernel.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  // check if complex_filter_kernel is 0-based, as currently required by fourier
  // TODO we could wrap-around if not
  for (int d=1; d<=num_dimensions; ++d)
    {
      if (min_indices[d]!=0) 
	return Succeeded::no;
    }
  max_indices[num_dimensions] = 2*max_indices[num_dimensions] - 1;
  padding_range = IndexRange<num_dimensions>(min_indices, max_indices);
  padded_sizes = max_indices - min_indices +1;
  return Succeeded::yes;
}

template <int num_dimensions, typename elemT>
Succeeded 
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_kernel(const Array<num_dimensions, elemT>& real_filter_kernel)  
{
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;
  if (!real_filter_kernel.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  // check if we need to use wrap-around
  if (norm(min_indices)<.01) // i.e. min_indices==0
    {
      complex_filter_kernel =
	fourier_for_real_data(real_filter_kernel);
    }
  else
    {
      // copy data to new kernel using wrap-around
      const BasicCoordinate<num_dimensions, int> sizes = 
	max_indices - min_indices + 1;
      const IndexRange<num_dimensions> range(sizes);
      Array<num_dimensions, elemT> real_filter_kernel_from_0(range);
      BasicCoordinate<num_dimensions, int> index = min_indices;
      do
	{
	  real_filter_kernel_from_0[modulo(index, sizes)] =
	    real_filter_kernel[index];
	}
      while(next(index, real_filter_kernel));

      // do DFT on this array
      complex_filter_kernel =
	fourier_for_real_data(real_filter_kernel_from_0);
    }

  return set_padding_range();
}

#ifndef __stir_ArrayFilterUsingRealDFTWithPadding_no_complex_kernel__
template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_kernel(const Array<num_dimensions, std::complex<elemT> >& complex_filter_kernel_v)
{
  complex_filter_kernel = complex_filter_kernel_v;
  return set_padding_range();
}
#endif

template <int num_dimensions, typename elemT>
bool ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
is_trivial() const
{
  return
    complex_filter_kernel.size_all()==0 ||
    (complex_filter_kernel.size_all()==1 && 
     (*complex_filter_kernel.begin_all()) == std::complex<elemT>(1,0));
}


template <int num_dimensions, typename elemT>
void 
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
do_it(Array<num_dimensions, elemT>& out_array, const Array<num_dimensions, elemT>& in_array) const
{
  if (in_array.get_index_range() == padding_range &&
      out_array.get_index_range() == padding_range)
    {
      // convolution using DFT
      {
	Array<num_dimensions,std::complex<elemT> > tmp =
	  fourier_for_real_data(in_array);
	tmp *= complex_filter_kernel;
	out_array = inverse_fourier_for_real_data(tmp);
      }
    }
    else
    {
      // copy input into padded_array using wrap-around
      Array<num_dimensions, elemT> padded_array(padding_range);
      assert(norm(get_min_indices(padded_array))<.01);// check padded_array is 0-based
      {
	BasicCoordinate<num_dimensions, int> index = get_min_indices(in_array);
	do
	  {
	    padded_array[modulo(index, padded_sizes)] = in_array[index];
	  }
	while(next(index, in_array));
      }
      // call do_it with padded_array
      do_it(padded_array, padded_array);
      // Now copy result in out_array using wrap-around
      {
	BasicCoordinate<num_dimensions, int> index = get_min_indices(out_array);
	do
	  {
	    out_array[index] = padded_array[modulo(index, padded_sizes)];
	  }
	while(next(index, out_array));
      }
    }
}

template class ArrayFilterUsingRealDFTWithPadding<1,float>;
template class ArrayFilterUsingRealDFTWithPadding<2,float>;
template class ArrayFilterUsingRealDFTWithPadding<3,float>;

END_NAMESPACE_STIR



