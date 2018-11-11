
#include "stir_experimental/SeparableGaussianArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/info.h"
#include <boost/format.hpp>

#include <iostream>
#include <fstream>

#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::fstream;
using std::iostream;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter()
:standard_deviation(0),number_of_coefficients(0)
{

 for (int i=1;i<=num_dimensions;i++)
  {
    this->all_1d_array_filters[i-1].
      reset(new ArrayFilter1DUsingConvolution<float>());
  }
}


template <int num_dimensions, typename elemT> 
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const float standard_deviation_v, 
                             const int number_of_coefficients_v)
			     :standard_deviation(standard_deviation_v),
			     number_of_coefficients(number_of_coefficients_v)

{
 
 VectorWithOffset<elemT> filter_coefficients;
 calculate_coefficients(filter_coefficients, number_of_coefficients_v,
			 standard_deviation_v);
 
 info("Printing filter coefficients - nonrescaled");
  for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)    
    info(boost::format("%1%   %2%   ") % i % filter_coefficients[i]);

  // rescaled to dc =1
 /* float sum =0.F;  
   for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)    
  { 
    sum +=double (filter_coefficients[i]);
  }
    
  cerr << " SUM IS " << sum << endl;

  for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)    
  { 
   filter_coefficients[i] /= sum;
  }

  cerr << " here  - rescaled" << endl;
   cerr << "Printing filter coefficients" << endl;
  for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)    
    cerr  << i<<"   "<< filter_coefficients[i] <<"   " << endl;
 */
  this->all_1d_array_filters[2].
    reset(new ArrayFilter1DUsingConvolution<float>(filter_coefficients));
  this->all_1d_array_filters[0].
    reset(new ArrayFilter1DUsingConvolution<float>());
  this->all_1d_array_filters[1].
    reset(new ArrayFilter1DUsingConvolution<float>(filter_coefficients));
  

}
template <int num_dimensions, typename elemT> 
void
SeparableGaussianArrayFilter<num_dimensions,elemT>:: 
calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients, const int number_of_coefficients,
			const float standard_deviation)

{

  filter_coefficients.grow(-number_of_coefficients,number_of_coefficients);
  filter_coefficients[0] = 1/sqrt(2*square(standard_deviation)*_PI);

  for (int i = 1; i<=number_of_coefficients;i++)
  { 
    filter_coefficients[i] = 
      filter_coefficients[-i]= 
      exp(-square(i)/(2.*square(standard_deviation)))/
      sqrt(2*square(standard_deviation)*_PI);
  }
    
}

template class SeparableGaussianArrayFilter<3,float>;

END_NAMESPACE_STIR
