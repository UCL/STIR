
#include "local/stir/SeparableGaussianArrayFilter.h"
#include "local/stir/ArrayFilter1DUsingConvolution.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"


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
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>();
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
 cerr << "Printing filter coefficients" << endl;
  for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)    
    cerr  << i<<"   "<< filter_coefficients[i] <<"   " << endl;

   all_1d_array_filters[2] = 	 
      new ArrayFilter1DUsingConvolution<float>(filter_coefficients);
   all_1d_array_filters[0] = 	 
       new ArrayFilter1DUsingConvolution<float>();
   all_1d_array_filters[1] = 	 
       new ArrayFilter1DUsingConvolution<float>(filter_coefficients);
  

}
template <int num_dimensions, typename elemT> 
void
SeparableGaussianArrayFilter<num_dimensions,elemT>:: 
calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients, const int number_of_coefficients,
			const float standard_deviation)

{

  filter_coefficients.grow(-number_of_coefficients,number_of_coefficients);
  filter_coefficients[0] = exp (-1/2*(square(1)/square(standard_deviation)));
  for (int i = 1; i<=number_of_coefficients;i++)
  { 
    filter_coefficients[i] = double(exp(double(-(square(i)/square(standard_deviation))/2)));
    filter_coefficients[-i]= double(exp(double(-(square(i)/square(standard_deviation))/2)));
  }
   cerr << " HERE " << endl;
  for (int i = 1; i<=number_of_coefficients;i++)
  {
    cerr << filter_coefficients[i] << "   " ;
  }
    
}

template SeparableGaussianArrayFilter<3,float>;

END_NAMESPACE_STIR