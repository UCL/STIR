//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ChainedImageProcessor

  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/
#include "tomo/ChainedImageProcessor.h"
#include "DiscretisedDensity.h"
#include <memory>

#ifndef TOMO_NO_NAMESPACES
using std::auto_ptr;
#endif

START_NAMESPACE_TOMO

  
template <int num_dimensions, typename elemT>
Succeeded
ChainedImageProcessor<num_dimensions, elemT>::
virtual_build_filter(const DiscretisedDensity<num_dimensions, elemT>& density)
{
  if (apply_first != 0)
    {
      // note that we cannot really build the filter for the 2nd 
      // as we don't know what the first will do to the dimensions etc. of the image
      return apply_first->build_filter(density);
    }
  else if (apply_second != 0)
    return apply_second->build_filter(density);
  else
    return Succeeded::yes;  
}


template <int num_dimensions, typename elemT>
void
ChainedImageProcessor<num_dimensions, elemT>::
virtual_apply(DiscretisedDensity<num_dimensions, elemT>& density) const
{  
  if (apply_first != 0)
    apply_first->apply(density);
  if (apply_second != 0)
    apply_second->apply(density);
}


template <int num_dimensions, typename elemT>
void
ChainedImageProcessor<num_dimensions, elemT>::
virtual_apply(DiscretisedDensity<num_dimensions, elemT>& out_density, 
	  const DiscretisedDensity<num_dimensions, elemT>& in_density) const
{
  if (apply_first != 0)
    {
      if (apply_second != 0)
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
      if (apply_second != 0)
	apply_second->apply(out_density, in_density);

}

template <int num_dimensions, typename elemT>
ChainedImageProcessor<num_dimensions, elemT>::
ChainedImageProcessor(ImageProcessor<num_dimensions,elemT> *const apply_first_v,
		      ImageProcessor<num_dimensions,elemT> *const apply_second_v)
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
}

template <int num_dimensions, typename elemT>
void 
ChainedImageProcessor<num_dimensions, elemT>::
initialise_keymap()
{
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

END_NAMESPACE_TOMO



