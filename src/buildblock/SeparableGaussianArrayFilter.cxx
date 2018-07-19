
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
:fwhm(0),max_kernel_sizes(0)
{

 for (int i=1;i<=num_dimensions;i++)
  {
    this->all_1d_array_filters[i-1].
      reset(new ArrayFilter1DUsingConvolution<float>());
  }
}

template <int num_dimensions, typename elemT>
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const float fwhm_v,const float max_kernel_sizes_v,  bool normalise)
:fwhm(fwhm_v),max_kernel_sizes(max_kernel_sizes_v)
    {

    //normalisation to 1 is optinal

        construct_filter(normalise);
    }


template <int num_dimensions, typename elemT> 
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const BasicCoordinate< num_dimensions,float>& fwhm_v,
                             const BasicCoordinate< num_dimensions,int>& max_kernel_sizes_v, bool normalise)

:fwhm(fwhm_v),max_kernel_sizes(max_kernel_sizes_v)
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
 for (int i = 1; i<=max_kernel_sizes.size();i++)

 {
    calculate_coefficients(filter_coefficients, max_kernel_sizes[i],
             fwhm[i],normalise);


    this->all_1d_array_filters[i].
    reset(new ArrayFilter1DUsingConvolution<float>(filter_coefficients));

   }
}

template <int num_dimensions, typename elemT> 
void
SeparableGaussianArrayFilter<num_dimensions,elemT>:: 
calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients, const int max_kernel_sizes,
            const float fwhm, bool normalise)

{

  float standard_deviation = sqrt(fwhm*fwhm/(8*log(2.F)));

  if (standard_deviation==0)
  {
      filter_coefficients.recycle();
      return;
  }

  const int kernel_lenght = max_kernel_sizes/2;
  filter_coefficients.grow(-kernel_lenght,kernel_lenght);

  filter_coefficients[0] = 1/sqrt(2*square(standard_deviation)*_PI);

  for (int i = 1; i<=kernel_lenght;i++)
  { 
    filter_coefficients[i] = 
      filter_coefficients[-i]= 
      exp(-square(i)/(2.*square(standard_deviation)))/
      sqrt(2*square(standard_deviation)*_PI);

// kernel.grow(0,2*max_kernel_size-1);


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
