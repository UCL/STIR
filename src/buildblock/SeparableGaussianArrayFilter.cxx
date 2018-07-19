
#include "stir/SeparableGaussianArrayFilter.h"
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
SeparableGaussianArrayFilter(const float standard_deviation_v,const float number_of_coefficients_v,  bool normalise)
:standard_deviation(standard_deviation_v),number_of_coefficients(number_of_coefficients_v)
    {

    //normalisation to 1 is optinal

        construct_filter(normalise);
    }


template <int num_dimensions, typename elemT> 
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const BasicCoordinate< num_dimensions,float>& standard_deviation_v,
                             const BasicCoordinate< num_dimensions,int>& number_of_coefficients_v, bool normalise)

:standard_deviation(standard_deviation_v),number_of_coefficients(0)
{
//normalisation to 1 is optinal

    construct_filter(normalise);
}


template <int num_dimensions, typename elemT>
void
SeparableGaussianArrayFilter<num_dimensions,elemT>::
construct_filter(bool normalise)
{
 VectorWithOffset<elemT> filter_coefficients;
 for (int i = 1; i<=number_of_coefficients.size();i++)

 {
    calculate_coefficients(filter_coefficients, number_of_coefficients[i],
             standard_deviation[i],normalise);


    this->all_1d_array_filters[i].
    reset(new ArrayFilter1DUsingConvolution<float>(filter_coefficients));

   }
}

template <int num_dimensions, typename elemT> 
void
SeparableGaussianArrayFilter<num_dimensions,elemT>:: 
calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients, const int number_of_coefficients,
            const float standard_deviation, bool normalise)

{
  if (standard_deviation==0)
  {
      filter_coefficients.recycle();
      return;
  }
  filter_coefficients.grow(-number_of_coefficients,number_of_coefficients);
  filter_coefficients[0] = 1/sqrt(2*square(standard_deviation)*_PI);

  for (int i = 1; i<=number_of_coefficients;i++)
  { 
    filter_coefficients[i] = 
      filter_coefficients[-i]= 
      exp(-square(i)/(2.*square(standard_deviation)))/
      sqrt(2*square(standard_deviation)*_PI);
  }


// normalisation: rescaled to dc =1

if (normalise)

    {

        float sum = 0.F;
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

    }
}


template class SeparableGaussianArrayFilter<3,float>;

END_NAMESPACE_STIR
