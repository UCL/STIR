//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class ThresholdMinToSmallPositiveValueImageProcessor

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/ThresholdMinToSmallPositiveValueImageProcessor.h"
#include "stir/recon_array_functions.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

  
template <typename elemT>
Succeeded
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)

{
  return Succeeded::yes;  
}


template <typename elemT>
void
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{     
  threshold_min_to_small_positive_value(density, rim_truncation_image);
}


template <typename elemT>
void
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  out_density = in_density;
  threshold_min_to_small_positive_value(out_density, rim_truncation_image);
}

template <typename elemT>
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
ThresholdMinToSmallPositiveValueImageProcessor(const int rim_truncation_image_v)
  : rim_truncation_image(rim_truncation_image_v)
{
  set_defaults();
}

template <typename elemT>
void
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
set_defaults()
{
  ImageProcessor<3, elemT>::set_defaults();
}

template <typename elemT>
void 
ThresholdMinToSmallPositiveValueImageProcessor<elemT>::
initialise_keymap()
{
  ImageProcessor<3, elemT>::initialise_keymap();
  this->parser.add_start_key("Threshold Min To Small Positive Value Parameters");
  this->parser.add_key("rim truncation (in pixels)", &rim_truncation_image);
  this->parser.add_stop_key("END Threshold Min To Small Positive Value Parameters");
}



const char * const 
ThresholdMinToSmallPositiveValueImageProcessor<float>::registered_name =
  "Threshold Min To Small Positive Value";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static ThresholdMinToSmallPositiveValueImageProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class ThresholdMinToSmallPositiveValueImageProcessor<float>;

END_NAMESPACE_STIR



