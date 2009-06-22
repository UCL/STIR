//
// $Id$
//
/*!

  \file
  \ingroup Array
  \brief Implementations for class stir::ArrayFilter1DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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

#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/IndexRange.h"
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

template <typename elemT>
ArrayFilter1DUsingConvolution<elemT>::
ArrayFilter1DUsingConvolution()
: filter_coefficients()
{
  
}

template <typename elemT>
ArrayFilter1DUsingConvolution<elemT>::
ArrayFilter1DUsingConvolution(const VectorWithOffset<elemT> &filter_coefficients_v)
: filter_coefficients(filter_coefficients_v)
{
  // TODO: remove 0 elements at the outside
}


template <typename elemT>
bool 
ArrayFilter1DUsingConvolution<elemT>::
is_trivial() const
{
  return
    filter_coefficients.get_length() == 0 ||
    (filter_coefficients.get_length()==1 && filter_coefficients.get_min_index()==0 &&
     filter_coefficients[0] == 1);
}


template <typename elemT>
Succeeded 
ArrayFilter1DUsingConvolution<elemT>::
get_influencing_indices(IndexRange<1>& influencing_index_range, 
                        const IndexRange<1>& input_index_range) const
{
  influencing_index_range = 
    (filter_coefficients.get_length() == 0)
    ? input_index_range
    : IndexRange<1>(input_index_range.get_min_index() - filter_coefficients.get_max_index(),
                    input_index_range.get_max_index() - filter_coefficients.get_min_index());
  return Succeeded::yes;
}

template <typename elemT>
Succeeded 
ArrayFilter1DUsingConvolution<elemT>:: 
get_influenced_indices(IndexRange<1>& influenced_index_range, 
                       const IndexRange<1>& input_index_range) const
{
  influenced_index_range = 
    (filter_coefficients.get_length() == 0)
    ? input_index_range
    : IndexRange<1>(input_index_range.get_min_index() + filter_coefficients.get_min_index(),
                    input_index_range.get_max_index() + filter_coefficients.get_max_index());
  return Succeeded::yes;
}

template <typename elemT>
void
ArrayFilter1DUsingConvolution<elemT>::
do_it(Array<1,elemT>& out_array, const Array<1,elemT>& in_array) const
{
  const int in_min = in_array.get_min_index();
  const int in_max = in_array.get_max_index();
  const int out_min = out_array.get_min_index();
  const int out_max = out_array.get_max_index();

  if (is_trivial())
  {    
    for (int i=out_min; i<=out_max; i++) 
    {
      out_array[i] = (i>=in_min && i <= in_max ? in_array[i] : 0);   
    }
    return;
  }
  const int j_min = filter_coefficients.get_min_index();
  const int j_max = filter_coefficients.get_max_index();

  for (int i=out_min; i<=out_max; i++) 
  {
    out_array[i] = 0;
    for (int j=max(j_min, i-in_max); j<=min(j_max, i-in_min); j++) 
      out_array[i] += filter_coefficients[j]*in_array[i-j];   
  }

}
// instantiation

template class ArrayFilter1DUsingConvolution<float>;

END_NAMESPACE_STIR

