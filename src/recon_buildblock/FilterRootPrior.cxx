//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
  \ingroup priors
  \brief  implementation of the stir::FilterRootPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic      
*/

#include "stir/recon_buildblock/FilterRootPrior.h"
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/DataProcessor.h"

START_NAMESPACE_STIR

template <typename DataT>
FilterRootPrior<DataT>::FilterRootPrior()
{
  set_defaults();
}


template <typename DataT>
FilterRootPrior<DataT>::
FilterRootPrior(shared_ptr<DataProcessor<DataT> >const& filter_sptr, float penalisation_factor_v)
:  filter_ptr(filter_sptr)
{
  this->penalisation_factor = penalisation_factor_v;
}


template < class T>
static inline int
sign (const T& x)
{ return x>=0 ? 1: -1;}

template < class T>
static inline T    // can't call this abs() as it overlaps with std::abs
my_abs(const T& x)
{ return x>=0 ? x: -x;}

/* A function that divides 2 floats while avoiding division by 0 by imposing an upper threshold
   It essentially returns 
     sign(numerator)*sign(denominator)*
       min(my_abs(numerator/denominator), max)
*/
template < class T>
static inline T
quotient_with_max(const T numerator, const T denominator, const T max)
{
  assert(max>0);
  return 
    my_abs(numerator)< max*my_abs(denominator) ? 
    numerator/denominator : 
    max * sign(numerator)*sign(denominator);
}

template <typename DataT>
double
FilterRootPrior<DataT>::
compute_value(const DataT &current_estimate)
{
  static bool first_time=true;
  if (first_time)
    {
      warning("FilterRootPrior:compute_value does not make sense. Just returning 0.");
      first_time=false;
    }
  return 0.;
}

template <typename DataT>
void 
FilterRootPrior<DataT>::
compute_gradient(DataT& prior_gradient, 
                 const DataT &current_image_estimate)
{
  assert(  prior_gradient.get_index_range() == current_image_estimate.get_index_range());  
  if (this->penalisation_factor==0 || filter_ptr==0)
  {
    std::fill(prior_gradient.begin_all(), prior_gradient.end_all(), 0);
    return;
  }

  this->check(current_image_estimate);
  

  // first store filtered image in prior_gradient
  filter_ptr->apply(prior_gradient,current_image_estimate);  

  /* now set 
     prior_gradient = current_image_estimate/filtered_image - 1
                    = current_image_estimate/prior_gradient - 1
     However, we need to avoid division by 0, as it might cause a NaN or an 'infinity'.
     (It seems that Intel processors handle 'infinity' alright, but sparc processors do not.)

     So, instead we do
     prior_gradient = quotient_with_max(current_image_estimate,prior_gradient,1000) - 1

     The code below does this by using a full_iterator loop as we're missing expression templates 
     at the moment and I did not feel like making a function object just for this ...
     */

  typename DataT::full_iterator iter_through_prior_gradient =
    prior_gradient.begin_all();
  typename DataT::const_full_iterator iter_through_current_image_estimate =
    current_image_estimate.begin_all();
  while (iter_through_current_image_estimate!= current_image_estimate.end_all())
  {
    *iter_through_prior_gradient=
      this->penalisation_factor * 
      (quotient_with_max(*iter_through_current_image_estimate,
			 *iter_through_prior_gradient, 
			 static_cast</*DataT::value_type*/float>(1000))
      - 1);
    ++iter_through_prior_gradient;
    ++iter_through_current_image_estimate;
  }
  assert(iter_through_prior_gradient == prior_gradient.end_all());
}



template <typename DataT>
void 
FilterRootPrior<DataT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("FilterRootPrior Parameters");
  this->parser.add_parsing_key("Filter type", &filter_ptr); 
  this->parser.add_stop_key("END FilterRootPrior Parameters");
}


template <typename DataT>
void
FilterRootPrior<DataT>::set_defaults()
{
  base_type::set_defaults();
  filter_ptr.reset();
}

template <typename DataT>
Succeeded
FilterRootPrior<DataT>::set_up (shared_ptr<DataT> const& target_sptr)
{
  base_type::set_up(target_sptr);

  return Succeeded::yes;
}

template <typename DataT>
void FilterRootPrior<DataT>::check(DataT const& current_image_estimate) const
{
  // Do base-class check
  base_type::check(current_image_estimate);
}

template <typename DataT>
const char * const 
FilterRootPrior<DataT>::
registered_name =
  "FilterRootPrior";

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class FilterRootPrior<DiscretisedDensity<3,float> >;
template class FilterRootPrior<ParametricVoxelsOnCartesianGrid >; 

END_NAMESPACE_STIR
