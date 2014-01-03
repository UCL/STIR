/*!
  \file
  \ingroup ImageProcessor  
  \brief Implementation of class stir::SeparableConvolutionImageFilter
    
  \author Kris Thielemans
  \author Sanida Mustafovic
*/
/*
    Copyright (C) 2002 - 2009-06-22, Hammersmith Imanet Ltd
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

#include "stir/SeparableConvolutionImageFilter.h"
#include "stir/SeparableArrayFunctionObject.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template<>
const char * const 
SeparableConvolutionImageFilter<float>::registered_name =
  "Separable Convolution";


template <typename elemT>
void 
SeparableConvolutionImageFilter<elemT>::
initialise_keymap()
{
  base_type::initialise_keymap();

  this->parser.add_start_key("Separable Convolution Filter Parameters");
  this->parser.add_key("x-dir filter coefficients", &(*(filter_coefficients_for_parsing.begin()+2)));
  this->parser.add_key("y-dir filter coefficients", &(*(filter_coefficients_for_parsing.begin()+1)));
  this->parser.add_key("z-dir filter coefficients", &(*filter_coefficients_for_parsing.begin()));
  this->parser.add_stop_key("END Separable Convolution Filter Parameters");

}

template <typename elemT>
bool 
SeparableConvolutionImageFilter<elemT>::
post_processing()
{
  if (base_type::post_processing() != false)
    return true;

  // copy filter_coefficients_for_parsing to filter_coefficients 
  // todo drop any 0s at the start or end

  typename VectorWithOffset< VectorWithOffset<elemT> >::iterator 
    coefficients_iter = filter_coefficients.begin();
#ifndef STIR_NO_NAMESPACES
      std::
#endif
  vector< vector<double> >::const_iterator 
    parsing_iter = filter_coefficients_for_parsing.begin();
  for (;
       parsing_iter != filter_coefficients_for_parsing.end();
       ++parsing_iter, ++coefficients_iter)
    {
      const unsigned int size = static_cast<unsigned int>(parsing_iter->size());
      const int min_index = -static_cast<int>((size/2));
      if (size%2==0)
	warning("Parsing SeparableConvolutionImageFilter\n"
		"Even number of filter coefficients for the %d-th dimension."
		"I'll (effectively) append a 0 at the end.\n",
		coefficients_iter - filter_coefficients.begin() + 1);

      *coefficients_iter = VectorWithOffset<elemT>(min_index, min_index + size - 1);

      // can't use std::copy because of cast. sigh.
      typename VectorWithOffset<elemT>::iterator 
	coefficients_elem_iter = coefficients_iter->begin();
#ifndef STIR_NO_NAMESPACES
      std::
#endif
	vector<double>::const_iterator 
	parsing_elem_iter = parsing_iter->begin();
      for (;
	   parsing_elem_iter != parsing_iter->end();
	   ++parsing_elem_iter, ++coefficients_elem_iter)
	*coefficients_elem_iter = static_cast<elemT>(*parsing_elem_iter);
    }

  return false;
}

template <typename elemT>
SeparableConvolutionImageFilter<elemT>::
SeparableConvolutionImageFilter()
: filter_coefficients_for_parsing(3)
{
    set_defaults();
}


template <typename elemT>
SeparableConvolutionImageFilter<elemT>::
SeparableConvolutionImageFilter(
				     const VectorWithOffset< VectorWithOffset<elemT> >&
				     filter_coefficients)
  : 
  filter_coefficients_for_parsing(filter_coefficients.get_length()),
  filter_coefficients(filter_coefficients)
{
  assert(filter_coefficients.get_length()==3);// num_dimensions

  // copy filter_coefficients to filter_coefficients_for_parsing such 
  // that get_parameters() works properly

  typename VectorWithOffset< VectorWithOffset<elemT> >::const_iterator 
    coefficients_iter = filter_coefficients.begin();
#ifndef STIR_NO_NAMESPACES
      std::
#endif
  vector< vector<double> >::iterator 
    parsing_iter = filter_coefficients_for_parsing.begin();
  for (;
       parsing_iter != filter_coefficients_for_parsing.end();
       ++parsing_iter, ++coefficients_iter)
    {
      // make sure that there are 0s appended such that parsing will read it back
      // in correct place
      const unsigned parsing_size = 
	2*(max(coefficients_iter->get_max_index(),
	       -coefficients_iter->get_min_index())) + 1;
      // make it long enough, and initialise with 0
      *parsing_iter = vector<double>(coefficients_iter->get_length(), 0);

      for (int i = coefficients_iter->get_min_index();
	   i <= coefficients_iter->get_max_index();
	   ++i)
	(*parsing_iter)[static_cast<unsigned>((parsing_size/2) + i)] = 
	  (*coefficients_iter)[i];
    }
      
}

#if 0
template <typename elemT>
VectorWithOffset<elemT>
SeparableConvolutionImageFilter<elemT>::
get_filter_coefficients()
{
  return filter_coefficients;
}
#endif

template <typename elemT>
Succeeded
SeparableConvolutionImageFilter<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > > 
    all_1d_filters(filter_coefficients.get_min_index(),
		   filter_coefficients.get_max_index());

  typename VectorWithOffset< VectorWithOffset<elemT> >::const_iterator 
    coefficients_iter = filter_coefficients.begin();
  typename VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::iterator 
    filter_iter = all_1d_filters.begin();
  for (;
       coefficients_iter != filter_coefficients.end();
       ++filter_iter, ++coefficients_iter)
    {
   
      filter_iter->reset(new ArrayFilter1DUsingConvolution<elemT>(*coefficients_iter));
    }  
  filter = SeparableArrayFunctionObject<3,elemT>(all_1d_filters);

  return Succeeded::yes;
  
}


template <typename elemT>
void
SeparableConvolutionImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{ 
  filter(density);  

}


template <typename elemT>
void
SeparableConvolutionImageFilter<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  filter(out_density,in_density);
}


template <typename elemT>
void
SeparableConvolutionImageFilter<elemT>::
set_defaults()
{
  base_type::set_defaults();
  filter_coefficients = 
      VectorWithOffset< VectorWithOffset<elemT> >(3);    
}


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class SeparableConvolutionImageFilter<float>;

END_NAMESPACE_STIR
