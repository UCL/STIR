//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ArrayFilter2DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ArrayFilter2DUsingConvolution.h"
#include "stir/IndexRange.h"
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

template <typename elemT>
ArrayFilter2DUsingConvolution<elemT>::
ArrayFilter2DUsingConvolution()
: filter_coefficients()
{
  
}

template <typename elemT>
ArrayFilter2DUsingConvolution<elemT>::
ArrayFilter2DUsingConvolution(const Array <2, float> &filter_coefficients_v)
: filter_coefficients(filter_coefficients_v)
{
  // TODO: remove 0 elements at the outside
}


template <typename elemT>
bool 
ArrayFilter2DUsingConvolution<elemT>::
is_trivial() const
{
  return
    filter_coefficients.get_length() == 0 ||
    (filter_coefficients.get_length()==1 && filter_coefficients.get_min_index()==0 &&
     filter_coefficients[0][0] == 1);
}


template <typename elemT>
Succeeded 
ArrayFilter2DUsingConvolution<elemT>::
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
ArrayFilter2DUsingConvolution<elemT>:: 
get_influenced_indices(IndexRange<1>& influenced_index_range, 
                       const IndexRange<1>& output_index_range) const
{
  influenced_index_range = 
    (filter_coefficients.get_length() == 0)
    ? output_index_range
    : IndexRange<1>(output_index_range.get_min_index() + filter_coefficients.get_min_index(),
                    output_index_range.get_max_index() + filter_coefficients.get_max_index());
  return Succeeded::yes;
}


template <typename elemT>
void
ArrayFilter2DUsingConvolution<elemT>::
do_it(Array<2,elemT>& out_array, const Array<2,elemT>& in_array) const
{
  const int in_min_y = in_array.get_min_index();
  const int in_max_y = in_array.get_max_index();
  const int in_min_x = in_array[in_min_y].get_min_index();
  const int in_max_x = in_array[in_min_y].get_max_index();
  
  
  const int out_min_y = out_array.get_min_index();
  const int out_max_y = out_array.get_max_index();
  const int out_min_x = out_array[out_min_y].get_min_index();
  const int out_max_x = out_array[out_min_y].get_max_index();
  

  
  if (is_trivial())
  {    
      for (int y=out_min_y; y<=out_max_y; y++) 
	for (int x=out_min_x; x<=out_max_x; x++) 
	  
	{
	  out_array[y][x] = ((y>=in_min_y && y <= in_max_y ) && 
	    (x>=in_min_x && x <= in_max_x ) ? in_array[y][x] : 0);   
	}
	return;
  }
  
  const int j_min = filter_coefficients.get_min_index();
  const int j_max = filter_coefficients.get_max_index();
  const int i_min = filter_coefficients[j_min].get_min_index();
  const int i_max = filter_coefficients[j_min].get_max_index();
  
  

 
    for (int y=out_min_y; y<=out_max_y; y++) 
      for (int x=out_min_x; x<=out_max_x; x++) 
      {
	out_array[y][x] = 0;
	
	  for (int j=max(j_min, y-in_max_y); j<=min(j_max, y-in_min_y); j++)  
	    for (int i=max(i_min, x-in_max_x); i<=min(i_max, x-in_min_x); i++) 
	      
	      out_array[y][x] += filter_coefficients[j][i]*in_array[y-j][x-i];   
      }
      
      
}


// instantiation

template class ArrayFilter2DUsingConvolution<float>;

END_NAMESPACE_STIR

