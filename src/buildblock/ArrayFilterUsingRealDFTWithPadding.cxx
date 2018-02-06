//
//
/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup Array
  \brief Implementation of class stir::ArrayFilterUsingRealDFTWithPadding

  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/ArrayFunction.h"
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
				   std::complex<elemT> >& kernel_in_frequency_space)
{ 
  if (set_kernel_in_frequency_space(kernel_in_frequency_space) == Succeeded::no) 
    error("Error constructing ArrayFilterUsingRealDFTWithPadding\n");
}
#endif

template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_padding_range()
{
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;

  if (!kernel_in_frequency_space.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  // check if kernel_in_frequency_space is 0-based, as currently required by fourier
  // TODO we could wrap-around if not
  for (int d=1; d<=num_dimensions; ++d)
    {
      if (min_indices[d]!=0) 
	return Succeeded::no;
    }
  max_indices[num_dimensions] = 2*max_indices[num_dimensions] - 1;
  this->padding_range = IndexRange<num_dimensions>(min_indices, max_indices);     
  this->padded_sizes = max_indices - min_indices +1;
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
      kernel_in_frequency_space =
	fourier_for_real_data(real_filter_kernel);
    }\
  else
    {
      // copy data to new kernel using wrap-around
      const BasicCoordinate<num_dimensions, int> sizes = 
	max_indices - min_indices + 1;
      const IndexRange<num_dimensions> range(sizes);
      Array<num_dimensions, elemT> real_filter_kernel_from_0(range);
      transform_array_to_periodic_indices(real_filter_kernel_from_0,
					  real_filter_kernel);

      // do DFT on this array
      kernel_in_frequency_space =
	fourier_for_real_data(real_filter_kernel_from_0);
    }

  return this->set_padding_range();
}

template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_kernel_in_frequency_space(const Array<num_dimensions, std::complex<elemT> >& kernel_in_frequency_space_v)
{
  kernel_in_frequency_space = kernel_in_frequency_space_v;
  return this->set_padding_range();
}

template <int num_dimensions, typename elemT>
bool ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
is_trivial() const
{
  return
    kernel_in_frequency_space.size_all()==0 ||
    (kernel_in_frequency_space.size_all()==1 && 
     (*kernel_in_frequency_space.begin_all()) == std::complex<elemT>(1,0));
}


template <int num_dimensions, typename elemT>
void 
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
do_it(Array<num_dimensions, elemT>& out_array, const Array<num_dimensions, elemT>& in_array) const
{
  if (in_array.get_index_range() == this->padding_range &&
      out_array.get_index_range() == this->padding_range)
    {
      // convolution using DFT
      {
	Array<num_dimensions,std::complex<elemT> > tmp =
	  fourier_for_real_data(in_array);
	tmp *= kernel_in_frequency_space;
	out_array = inverse_fourier_for_real_data_corrupting_input(tmp);
      }
    }
    else
    {
      // copy input into padded_array using wrap-around

      Array<num_dimensions, elemT> padded_array(this->padding_range);
      transform_array_to_periodic_indices(padded_array, in_array);
      // call do_it with padded_array
      do_it(padded_array, padded_array);
      // Now copy result in out_array using wrap-around
      transform_array_from_periodic_indices(out_array, padded_array);
    }
}

template class ArrayFilterUsingRealDFTWithPadding<1,float>;
template class ArrayFilterUsingRealDFTWithPadding<2,float>;
template class ArrayFilterUsingRealDFTWithPadding<3,float>;

END_NAMESPACE_STIR



