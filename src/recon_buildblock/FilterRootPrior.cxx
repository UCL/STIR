//
// $Id$
//
/*!
  \file
  \ingroup priors
  \brief  implementation of the FilterRootPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic      
  $Date$        
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/FilterRootPrior.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ImageProcessor.h"

START_NAMESPACE_STIR

template <typename elemT>
FilterRootPrior<elemT>::FilterRootPrior()
{
  set_defaults();
}


template <typename elemT>
FilterRootPrior<elemT>::FilterRootPrior(ImageProcessor<3,elemT>* filter, float penalisation_factor_v)
:  filter_ptr(filter)
{
  this->penalisation_factor = penalisation_factor_v;
}


static inline int
sign (const float x)
{ return x>=0 ? 1: -1;}

/* A function that divides 2 floats while avoiding division by 0 by imposing an upper threshold
   It essentially returns 
     sign(numerator)*sign(denominator)*
       min(fabs(numerator/denominator), max)
*/
static inline float
quotient_with_max(const float numerator, const float denominator, const float max)
{
  assert(max>0);
  return 
    fabs(numerator)< max*fabs(denominator) ? 
    numerator/denominator : 
    max * sign(numerator)*sign(denominator);
}

template <typename elemT>
void 
FilterRootPrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
                 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_gradient.get_index_range() == current_image_estimate.get_index_range());  
  if (this->penalisation_factor==0 || filter_ptr==0)
  {
    prior_gradient.fill(0);
    return;
  }
  

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

  typename DiscretisedDensity<3,elemT>::full_iterator iter_through_prior_gradient =
    prior_gradient.begin_all();
  typename DiscretisedDensity<3,elemT>::const_full_iterator iter_through_current_image_estimate =
    current_image_estimate.begin_all();
  while (iter_through_current_image_estimate!= current_image_estimate.end_all())
  {
    *iter_through_prior_gradient=
      this->penalisation_factor * 
      (quotient_with_max(*iter_through_current_image_estimate,*iter_through_prior_gradient, 1000)
      - 1);
    ++iter_through_prior_gradient;
    ++iter_through_current_image_estimate;
  }
  assert(iter_through_prior_gradient == prior_gradient.end_all());
}



template <typename elemT>
void 
FilterRootPrior<elemT>::initialise_keymap()
{
  GeneralisedPrior<elemT>::initialise_keymap();
  this->parser.add_start_key("FilterRootPrior Parameters");
  this->parser.add_parsing_key("Filter type", &filter_ptr); 
  this->parser.add_stop_key("END FilterRootPrior Parameters");
}


template <typename elemT>
void
FilterRootPrior<elemT>::set_defaults()
{
  GeneralisedPrior<elemT>::set_defaults();
  filter_ptr = 0;  
}

template <>
const char * const 
FilterRootPrior<float>::registered_name =
  "FilterRootPrior";

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


#if 0
// registration stuff moved to recon_buildblock_registries.cxx

// Register this class in the ImageProcessor registry
static FilterRootPrior<float>::RegisterIt dummy;
#endif

template class FilterRootPrior<float>;

END_NAMESPACE_STIR

