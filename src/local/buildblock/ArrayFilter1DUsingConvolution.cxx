//
// $Id$
//
/*!

  \file

  \brief Implementations for class ArrayFilter1DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#include "local/tomo/ArrayFilter1DUsingConvolution.h"

START_NAMESPACE_TOMO

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
    (filter_coefficients.get_length()==1 &&
     filter_coefficients[filter_coefficients.get_min_index()] == 1);
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

template ArrayFilter1DUsingConvolution<float>;

END_NAMESPACE_TOMO

