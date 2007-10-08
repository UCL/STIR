//
// $Id$
//
/*
    Copyright (C) 2007- $Date$, Hammersmith Imanet
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
  \brief Declaration of class NonseparableConvolutionUsingRealDFTImageFilter
    
  \author Kris Thielemans
  \author Charalampos Tsoumpas
      
  $Date$
  $Revision$
*/


#ifndef __stir_NonseparableConvolutionUsingRealDFTImageFilter_H__
#define __stir_NonseparableConvolutionUsingRealDFTImageFilter_H__


#include "stir/NonseparableConvolutionUsingRealDFTImageFilter.h"
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VectorWithOffset.h"
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif


START_NAMESPACE_STIR

// TODO!! remove define
// currently fixed at 3 because I didn't really have a good idea for the parsing 
// keywords in n dimensions.
//
#define num_dimensions 3

/*!
  \ingroup ImageProcessor  
  \brief A class derived from DataProcessor for performing separable
    periodic convolutions with an array kernel.

    This filter applies a 3D convolution based on the filter kernel.

    When parsing, the filter coefficients are read as an Array

    Convolution is periodic. 

    Elements of the input array that are outside its
    index range are considered to be 0.   

    \warning There is NO check if the kernel coefficients add up to 1. This is 
    because not all filters need this (e.g. edge enhancing filters).

    \par Example input for a kernel
    \verbatim
    Nonseparable Convolution Using Real DFT Image Filter :=
    filter kernel := kernel.hv 
    END Nonseparable Convolution Using Real DFT Image Filter :=
    \endverbatim

    The filter is implemented using the class ArrayFilterUsingRealDFTWithPadding.
*/
template <typename elemT>
class NonseparableConvolutionUsingRealDFTImageFilter : 
  public 
    RegisteredParsingObject<
        NonseparableConvolutionUsingRealDFTImageFilter<elemT>,
        DataProcessor<DiscretisedDensity<3,elemT> >,
        DataProcessor<DiscretisedDensity<3,elemT> >
    >
{
 private:
  typedef
    RegisteredParsingObject<
              NonseparableConvolutionUsingRealDFTImageFilter<elemT>,
              DataProcessor<DiscretisedDensity<3,elemT> >,
              DataProcessor<DiscretisedDensity<3,elemT> >
	       >
    base_type;
public:
  //! Name for parsing registry
  static const char * const registered_name; 
  
  //! Default constructor
  NonseparableConvolutionUsingRealDFTImageFilter();

  //! Constructor taking filter kernel explicitly.
  /*! This kernel is passed to the 
      ArrayFilterUsingRealDFTWithPadding constructor.

  */
  NonseparableConvolutionUsingRealDFTImageFilter( const Array<num_dimensions,elemT>& filter_coefficients ); 
  
private:
  string _kernel_filename;
  shared_ptr<DiscretisedDensity<num_dimensions,elemT> > _kernel_sptr;
  shared_ptr<ArrayFilterUsingRealDFTWithPadding<num_dimensions,elemT> > _array_filter_sptr; // ChT::float
  Array<num_dimensions,elemT> _filter_coefficients;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
  Succeeded virtual_set_up(const DiscretisedDensity<num_dimensions,elemT>& image);
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& out_density, 
		      const DiscretisedDensity<num_dimensions,elemT>& in_density) const;
  void  virtual_apply(DiscretisedDensity<num_dimensions,elemT>& density) const ;
  
};

#undef num_dimensions

END_NAMESPACE_STIR

#endif


