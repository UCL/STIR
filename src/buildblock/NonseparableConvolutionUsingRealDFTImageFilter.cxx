/*
    Copyright (C) 2007 - 2007-10-08, Hammersmith Imanet
    Copyright (C) 2011, Kris Thielemans
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

/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class stir::NonseparableConvolutionUsingRealDFTImageFilter
  
  \author Sanida Mustafovic
  \author Kris Thielemans
*/

#include "stir/NonseparableConvolutionUsingRealDFTImageFilter.h"
#include "stir/IndexRange3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/ArrayFunction.h"
#include "stir/Array_complex_numbers.h"
#include "stir/IO/read_from_file.h"


START_NAMESPACE_STIR

// fixed to 3D at present
static const int num_dimensions = 3;

template <typename elemT>
NonseparableConvolutionUsingRealDFTImageFilter<elemT>::
NonseparableConvolutionUsingRealDFTImageFilter()
{ 
  this->set_defaults();
}

template <typename elemT>
NonseparableConvolutionUsingRealDFTImageFilter<elemT>::
NonseparableConvolutionUsingRealDFTImageFilter( const Array<3,elemT>& filter_coefficients )      
{
  this->_filter_coefficients = filter_coefficients;  
}

template <typename elemT>
Succeeded 
NonseparableConvolutionUsingRealDFTImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{  
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;
  if (!density.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  BasicCoordinate<num_dimensions, int> padded_sizes = max_indices - min_indices +1;
  if (!this->_filter_coefficients.get_regular_range(min_indices, max_indices))
    return Succeeded::no;
  padded_sizes += max_indices - min_indices +1;
  // remove 1 to be accurate
 padded_sizes -= 1;
  // need to make it a power of 2 for the DFT implementation
  for (int d=1; d<= num_dimensions; ++d)
    {
      padded_sizes[d] =
	static_cast<int>(round(pow(2., ceil(log(static_cast<double>(padded_sizes[d])) / log(2.)))));
    }
  IndexRange<num_dimensions> padding_range(padded_sizes);
  Array<num_dimensions, elemT> padded_filter_coefficients(padding_range);
  transform_array_to_periodic_indices(padded_filter_coefficients, this->_filter_coefficients);
  this->_array_filter_sptr.reset(
				 new ArrayFilterUsingRealDFTWithPadding<num_dimensions,elemT>(padded_filter_coefficients));
  return Succeeded::yes;       
}

template <typename elemT>
void
NonseparableConvolutionUsingRealDFTImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const
{
  (*this->_array_filter_sptr)(out_density, in_density);
}

template <typename elemT>
void
NonseparableConvolutionUsingRealDFTImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& density) const
{
  // should use scoped_ptr but don't have it yet
  shared_ptr<DiscretisedDensity<3,elemT> > tmp_density_sptr(density.clone());
  this->virtual_apply(density, *tmp_density_sptr);
}


template <typename elemT>
void
NonseparableConvolutionUsingRealDFTImageFilter<elemT>::set_defaults()
{
  this->_kernel_filename="";
  this->_filter_coefficients.fill(0.F);
}
     
template <typename elemT>
void
NonseparableConvolutionUsingRealDFTImageFilter<elemT>:: initialise_keymap()
{
  this->parser.add_start_key("Nonseparable Convolution Using Real DFT Image Filter");
  this->parser.add_key("filter kernel", &this->_kernel_filename);
  this->parser.add_stop_key("END Nonseparable Convolution Using Real DFT Image Filter");
}
     
template <typename elemT>
bool 
NonseparableConvolutionUsingRealDFTImageFilter<elemT>::
post_processing()
{
  if (this->_kernel_filename.length() == 0)
    { warning("You need to specify a kernel file"); return true; }
  else
    this->_kernel_sptr = read_from_file<DiscretisedDensity<3,elemT> >(this->_kernel_filename);
  const  DiscretisedDensity<3,elemT>& kernel = *this->_kernel_sptr;
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;
  if (!kernel.get_regular_range(min_indices, max_indices))
    return true;

  const BasicCoordinate<num_dimensions, int> sizes = max_indices - min_indices + 1;
  
  if (sizes[1]%2==0 || sizes[2]%2==0 || sizes[3]%2==0)
	warning("Parsing Nonseparable Convolution Using Real DFT Image Filter\n"
		"Even number of filter coefficients for at least one of the dimensions."
		"I'll (effectively) append a 0 at the end.\n");

  const BasicCoordinate<num_dimensions, int> new_min_indices = (sizes/2)*(-1);

  this->_filter_coefficients.grow(IndexRange<num_dimensions>(new_min_indices, new_min_indices + sizes - 1 ));
 
  BasicCoordinate<num_dimensions, int> index = get_min_indices(this->_filter_coefficients);
  do
    {
      this->_filter_coefficients[index] = kernel[index - new_min_indices + min_indices];
    }
  while(next(index, this->_filter_coefficients));

  return false;
}

template<>    
const char * const 
NonseparableConvolutionUsingRealDFTImageFilter<float>::registered_name =
"Nonseparable Convolution Using Real DFT Image Filter";
     
#  ifdef _MSC_VER
     // prevent warning message on reinstantiation, 
     // note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class NonseparableConvolutionUsingRealDFTImageFilter<float>;
 
END_NAMESPACE_STIR
       
