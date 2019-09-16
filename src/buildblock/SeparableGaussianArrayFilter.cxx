//
//
/*!

  \file
  \ingroup Array

  \brief Implementations for class stir::SeparableGaussianArrayFilter

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Ludovica Brusaferri

*/
/*
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2018, UCL
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

#include "stir/SeparableGaussianArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/VectorWithOffset.h"
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
:fwhms(0),max_kernel_sizes(0)
{
 for (int i=1;i<=num_dimensions;i++)

  {
    this->all_1d_array_filters[i-1].
      reset(new ArrayFilter1DUsingConvolution<float>());
  }
}

template <int num_dimensions, typename elemT>
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const float fwhms_v,const float max_kernel_sizes_v,  bool normalise)
:fwhms(fwhms_v),max_kernel_sizes(max_kernel_sizes_v)
    {

    //normalisation to 1 is optinal
        construct_filter(normalise);
    }


template <int num_dimensions, typename elemT> 
SeparableGaussianArrayFilter<num_dimensions,elemT>::
SeparableGaussianArrayFilter(const BasicCoordinate< num_dimensions,float>& fwhms_v,
                             const BasicCoordinate< num_dimensions,int>& max_kernel_sizes_v, bool normalise)

:fwhms(fwhms_v),max_kernel_sizes(max_kernel_sizes_v)
{
    construct_filter(normalise);
}


template <int num_dimensions, typename elemT>
void
SeparableGaussianArrayFilter<num_dimensions,elemT>::
construct_filter(bool normalise)
{
 VectorWithOffset<elemT> filter_coefficients;
 for (int i = 1; i<=num_dimensions;i++)

 {
    calculate_coefficients(filter_coefficients, max_kernel_sizes[i],
             fwhms[i],normalise);

    this->all_1d_array_filters[i-1].
    reset(new ArrayFilter1DUsingConvolution<float>(filter_coefficients));

   }
}

template <int num_dimensions, typename elemT> 
void
SeparableGaussianArrayFilter<num_dimensions,elemT>:: 
calculate_coefficients(VectorWithOffset<elemT>& filter_coefficients, const int max_kernel_sizes,
            const float fwhms, bool normalise)

{

  float standard_deviation = sqrt(fwhms*fwhms/(8*log(2.F)));

  if (standard_deviation==0)

  {
      filter_coefficients.recycle();
      return;
  }

  const int kernel_length = max_kernel_sizes/2;
  filter_coefficients.grow(-kernel_length,kernel_length);

  filter_coefficients[0] = static_cast<elemT>(1/sqrt(2*square(standard_deviation)*_PI));

  for (int i = 1; i<=kernel_length;i++)
  { 
    filter_coefficients[i] = 
      filter_coefficients[-i]= static_cast<elemT>(
      exp(-square(i)/(2.*square(standard_deviation)))/
      sqrt(2*square(standard_deviation)*_PI));

// kernel.grow(0,2*max_kernel_size-1);


  }


// normalisation: rescaled to dc =1

if (normalise)

    {

        double sum = 0.;
        for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)
        {
          sum +=double (filter_coefficients[i]);
        }

        for (int i =filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)
        {
         filter_coefficients[i] /= sum;
        }

    }

}


template class SeparableGaussianArrayFilter<3,float>;

END_NAMESPACE_STIR
