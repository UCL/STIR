//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class TruncateMinToSmallPositiveValueImageProcessor

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/
#include "tomo/TruncateMinToSmallPositiveValueImageProcessor.h"
#include "recon_array_functions.h"
#include "DiscretisedDensity.h"

START_NAMESPACE_TOMO

  
template <typename elemT>
Succeeded
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)

{
  return Succeeded::yes;  
}


template <typename elemT>
void
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{     
  truncate_min_to_small_positive_value(density, rim_truncation_image);
}


template <typename elemT>
void
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  out_density = in_density;
  truncate_min_to_small_positive_value(out_density, rim_truncation_image);
}

template <typename elemT>
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
TruncateMinToSmallPositiveValueImageProcessor(const int rim_truncation_image_v)
  : rim_truncation_image(rim_truncation_image_v)
{
  set_defaults();
}

template <typename elemT>
void
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
set_defaults()
{
}

template <typename elemT>
void 
TruncateMinToSmallPositiveValueImageProcessor<elemT>::
initialise_keymap()
{
  parser.add_start_key("Truncate Min To Small Positive Value Parameters");
  parser.add_key("rim truncation (in pixels)", &rim_truncation_image);
  parser.add_stop_key("END Truncate Min To Small Positive Value Parameters");
}



const char * const 
TruncateMinToSmallPositiveValueImageProcessor<float>::registered_name =
  "Truncate Min To Small Positive Value";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static TruncateMinToSmallPositiveValueImageProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template TruncateMinToSmallPositiveValueImageProcessor<float>;

END_NAMESPACE_TOMO



