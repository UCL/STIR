//
//
/*!

  \file
  \ingroup Array
  \brief Implementations for class stir::ArrayFilter1DUsingConvolutionSymmetricKernel

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

#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"

#include <algorithm>
using std::min;

START_NAMESPACE_STIR
template <typename elemT>
ArrayFilter1DUsingConvolutionSymmetricKernel<elemT>::
ArrayFilter1DUsingConvolutionSymmetricKernel(const VectorWithOffset<elemT> &filter_coefficients_v)
: filter_coefficients(filter_coefficients_v)
{
  // TODO: remove 0 elements at the outside

  assert(filter_coefficients.get_length() == 0 ||
         filter_coefficients.get_min_index()==0);
}


template <typename elemT>
bool 
ArrayFilter1DUsingConvolutionSymmetricKernel<elemT>::
is_trivial() const
{
  return
    filter_coefficients.get_length() == 0 ||
    (filter_coefficients.get_length()==1 &&
     filter_coefficients[0] == 1);
}

// TODO generalise to arbitrary index ranges
template <typename elemT>
void
ArrayFilter1DUsingConvolutionSymmetricKernel<elemT>::
do_it(Array<1,elemT>& out_array, const Array<1,elemT>& in_array) const
{
  assert(out_array.get_min_index() == in_array.get_min_index());
  assert(out_array.get_max_index() == in_array.get_max_index());

  if (is_trivial())
  {
    out_array = in_array;
    return;
  }

  const int in_min = in_array.get_min_index();
  const int in_max = in_array.get_max_index();
  const int j_max = filter_coefficients.get_max_index();

  for (int i=in_array.get_min_index(); 
       i<=in_array.get_max_index(); i++) 
  {
    out_array[i] = filter_coefficients[0]*in_array[i];
    int j=1;
    // first do range where both i-j and i+j indices are valid for in_array
    for (; j<=min(j_max,min(in_max-i, i-in_min)); j++) 
      out_array[i] += filter_coefficients[j]*(in_array[i-j]+in_array[i+j]);  
    //if (j>j_max) return;

    // now do rest of i+j separate
    // next conditional is not necessary
    //if (i-in_min < in_max-i)
    {
      for (; j<=min(j_max,in_max-i); j++) 
        out_array[i] += filter_coefficients[j]*(in_array[i+j]);    
    }
    // now do rest of i-j separate
    // next conditional is not necessary
    //else
    {
      for (; j<=min(j_max,i-in_min); j++) 
        out_array[i] += filter_coefficients[j]*(in_array[i-j]);    
    }
  }
#if 0
  // untested implementation using padding
  VectorWithOffset<elemT> in_array_padded = in_array;
  // next padding statement is bound to be problematic (for instance, take positive min_index)
  in_array_padded.grow(2*in_array.get_min_index(),2*in_array.get_max_index());
  in_array_padded.fill(0);

  for (int i= in_array.get_min_index();i<=in_array.get_max_index();i++)
  in_array_padded[i] = in_array[i];
  int i,k ;

  // make a vector (2*in_array.get_min_index(),2*in_array.get_max_index()
  VectorWithOffset<float> tmp_cfilter =in_array_padded;
  
  int length = in_array.get_length();
  int flen =  filter_coefficients.get_length();
 
  tmp_cfilter.fill(0);
  int r2=flen-1;
  for(int r1=(in_array.get_min_index()-flen+1);r1<in_array.get_min_index()+1;r1++)
  {
      tmp_cfilter[r1] = filter_coefficients[r2];
      r2--;
  }
 
  VectorWithOffset<float> multiplyed =tmp_cfilter ;
  multiplyed.fill(0);
  VectorWithOffset<float> shifted = tmp_cfilter;
 
  float sum =0;
  VectorWithOffset<float>out_array_tmp;

  // output -> data+filter-1
  out_array_tmp.grow(shifted.get_min_index(),shifted.get_max_index()+flen-3);
  
  out_array_tmp.fill(0);
  VectorWithOffset<float> shifted_tmp = shifted;
  
  for (int j=out_array_tmp.get_min_index(); j<=out_array_tmp.get_max_index(); j++) 
  {   
    for(int i=shifted_tmp.get_min_index();i<shifted_tmp.get_max_index();i++)
    {     
      multiplyed[i] = shifted_tmp[i]*in_array_padded[i];
      sum += multiplyed[i] ;
    }
    out_array_tmp[j] =sum;
    sum =0;
    shifted = shifted_tmp;
    cir_shift_to_right(shifted_tmp,shifted);    
  } 
  int f1 = out_array_tmp.get_min_index();
  for (int f =in_array.get_min_index();f<=in_array.get_max_index();f++)
  {
    out_array[f] = out_array_tmp[f1];
    f1++;
  }
#endif
}


#if 0
template <typename elemT>
static void
cir_shift_to_right(VectorWithOffset<elemT>&output,const VectorWithOffset<elemT>&input)
{
  output[input.get_min_index()]= input[input.get_max_index()];
  for (int i=input.get_min_index();i<=input.get_max_index()-1;i++)    
    output[i+1]= input[i];
}

#endif

// instantiation

template class ArrayFilter1DUsingConvolutionSymmetricKernel<float>;

END_NAMESPACE_STIR

