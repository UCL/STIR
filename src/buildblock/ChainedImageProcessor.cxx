//
// $Id$
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class ChainedImageProcessor

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ChainedImageProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include <memory>

#ifndef STIR_NO_NAMESPACES
using std::auto_ptr;
#endif

START_NAMESPACE_STIR

  
template <int num_dimensions, typename elemT>
Succeeded
ChainedImageProcessor<num_dimensions, elemT>::
virtual_set_up(const DiscretisedDensity<num_dimensions, elemT>& density)
{
  if (!is_null_ptr(apply_first))
    {
      // note that we cannot really build the filter for the 2nd 
      // as we don't know what the first will do to the dimensions etc. of the image
      return apply_first->set_up(density);
    }
  else if (!is_null_ptr(apply_second))
    return apply_second->set_up(density);
  else
    return Succeeded::yes;  
}


template <int num_dimensions, typename elemT>
void
ChainedImageProcessor<num_dimensions, elemT>::
virtual_apply(DiscretisedDensity<num_dimensions, elemT>& density) const
{  
  if (!is_null_ptr(apply_first))
    apply_first->apply(density);
  if (!is_null_ptr(apply_second))
    apply_second->apply(density);
}


template <int num_dimensions, typename elemT>
void
ChainedImageProcessor<num_dimensions, elemT>::
virtual_apply(DiscretisedDensity<num_dimensions, elemT>& out_density, 
	  const DiscretisedDensity<num_dimensions, elemT>& in_density) const
{
  if (!is_null_ptr(apply_first))
    {
      if (!is_null_ptr(apply_second))
	{
	  // a bit complicated because we need a temporary image
	  auto_ptr< DiscretisedDensity<num_dimensions, elemT> > temp_density_ptr =
	    auto_ptr< DiscretisedDensity<num_dimensions, elemT> >(
		   in_density.get_empty_discretised_density()
		   );      
	  apply_first->apply(*temp_density_ptr, in_density);
	  apply_second->apply(out_density, *temp_density_ptr);
	}
      else
	apply_first->apply(out_density, in_density);
    }
  else
      if (!is_null_ptr(apply_second))
	apply_second->apply(out_density, in_density);

}

template <int num_dimensions, typename elemT>
ChainedImageProcessor<num_dimensions, elemT>::
ChainedImageProcessor(shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_first_v,
		      shared_ptr<ImageProcessor<num_dimensions,elemT> > const& apply_second_v)
  : apply_first(apply_first_v),
    apply_second(apply_second_v)
{
  set_defaults();
}

template <int num_dimensions, typename elemT>
void
ChainedImageProcessor<num_dimensions, elemT>::
set_defaults()
{
  ImageProcessor<num_dimensions, elemT>::set_defaults();
}

template <int num_dimensions, typename elemT>
void 
ChainedImageProcessor<num_dimensions, elemT>::
initialise_keymap()
{
  ImageProcessor<num_dimensions, elemT>::initialise_keymap();
  parser.add_start_key("Chained Image Processor Parameters");
  parser.add_parsing_key("Image Processor to apply first", &apply_first);
  parser.add_parsing_key("Image Processor to apply second", &apply_second);
  parser.add_stop_key("END Chained Image Processor Parameters");
}



const char * const 
ChainedImageProcessor<3,float>::registered_name =
  "Chained Image Processor";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static ChainedImageProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template ChainedImageProcessor<3,float>;

END_NAMESPACE_STIR



