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

#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/numerics/fourier.h"
#include "stir/Succeeded.h"
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
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

template <int num_dimensions, typename elemT>
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
ArrayFilterUsingRealDFTWithPadding(const Array<num_dimensions, 
				               std::complex<elemT> >& complex_filter_kernel)
{ 
  if (set_kernel(complex_filter_kernel) == Succeeded::no) 
    error("Error constructing ArrayFilterUsingRealDFTWithPadding\n");
}

template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_padding_range()
{
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;

  if (!complex_filter_kernel.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  for (int d=1; d<=num_dimensions; ++d)
    {
      if (min_indices[d]!=0) // TODO currently required by fourier
	return Succeeded::no;
      max_indices[d] = 2*max_indices[d] - 1;
    }
  padding_range = IndexRange<num_dimensions>(min_indices, max_indices);
  return Succeeded::yes;
}

template <int num_dimensions, typename elemT>
Succeeded 
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_kernel(const Array<num_dimensions, elemT>& real_filter_kernel)  
{
  // check if we need to use wrap-around
  // note: this check only works in 1D (TODO)
  if (real_filter_kernel.get_min_index()==0)
    {
      complex_filter_kernel =
	fourier_1d_for_real_data(real_filter_kernel);
    }
  else
    {
      Array<num_dimensions, elemT> real_filter_kernel_from_0(real_filter_kernel.get_index_range());
      const int length=static_cast<int>(real_filter_kernel.size());
      real_filter_kernel_from_0.set_min_index(0);
      for (int i=real_filter_kernel.get_min_index(); i<= real_filter_kernel.get_max_index(); ++i)
	real_filter_kernel_from_0[modulo(i,length)] =
	  real_filter_kernel[i];
      complex_filter_kernel =
	fourier_1d_for_real_data(real_filter_kernel_from_0);
    }


  return set_padding_range();
}

template <int num_dimensions, typename elemT>
Succeeded
ArrayFilterUsingRealDFTWithPadding<num_dimensions, elemT>:: 
set_kernel(const Array<num_dimensions, std::complex<elemT> >& complex_filter_kernel_v)
{
  complex_filter_kernel = complex_filter_kernel_v;
  return set_padding_range();
}


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
  Array<num_dimensions, elemT> padded_array(padding_range);
  // TODO this code only works for 1D at present
  if (in_array.size()>padded_array.size())
    error("ArrayFilterUsingRealDFTWithPadding called with in_array which is too long for the filter kernel");
  std::copy(in_array.begin(), in_array.end(), padded_array.begin());

  {
    Array<num_dimensions,std::complex<elemT> > tmp =
      fourier_1d_for_real_data(padded_array);
    tmp *= complex_filter_kernel;
    padded_array = inverse_fourier_1d_for_real_data(tmp);
  }
  // Now copy result in out_array
  // Note that padded_array[0] corresponds to out_array[in_array.get_min_index()].
  // In case out_array is longer at the 'left' or 'right', use wrap-around in padded_array
  assert(padded_array.get_min_index()==0);
#if 0
  // For padded_array, 0 corresponds to corresponds to padded_array.get_max_index()+1
  // this probably works, but assumes that out_array is not too much longer than in_array
  for (int i=out_array.get_min_index(); i<in_array.get_min_index() && i<=out_array.get_max_index(); ++i)
    out_array[i] = padded_array[padded_array.get_max_index()+1+(i-in_array.get_min_index())];
  for (int i=in_array.get_min_index(); i<out_array.get_min_index(); ++i)
    out_array[i] = padded_array[i-in_array.get_min_index()];
#else
  const int length= static_cast<int>(padded_array.size());
  for (int i=out_array.get_min_index(), 
	 i_padded=out_array.get_min_index()-in_array.get_min_index(); 
       i<=out_array.get_max_index(); 
       ++i, ++i_padded)
    {
      out_array[i] = padded_array[modulo(i_padded, length)];
    }
#endif
}

template class ArrayFilterUsingRealDFTWithPadding<1,float>;

END_NAMESPACE_STIR



