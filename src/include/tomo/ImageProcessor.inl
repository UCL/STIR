//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for class ImageProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/
#if 0
// lines necessary for .cxx, but it's now .inl
#include "tomo/ImageProcessor.h"

#ifdef _MSC_VER
// disable warnings on pure virtuals
#pragma warning(disable: 4661)
#endif // _MSC_VER

#endif // 0


START_NAMESPACE_TOMO

 
template <int num_dimensions, typename elemT>
ImageProcessor<num_dimensions,elemT>::
ImageProcessor()
: filter_built(false)
{}
   
template <int num_dimensions, typename elemT>
Succeeded 
ImageProcessor<num_dimensions,elemT>::
build_filter(const DiscretisedDensity< num_dimensions,elemT>& image)
{
  Succeeded result = virtual_build_filter(image);
  filter_built = (result == Succeeded::yes);
  return result;
}


template <int num_dimensions, typename elemT>
void 
ImageProcessor<num_dimensions,elemT>::
apply(DiscretisedDensity<num_dimensions,elemT>& density)
  {
    //assert(consistency_check(density) == Succeeded::yes);
    if (!filter_built )
      build_filter(density);
    virtual_apply(density);
  }


template <int num_dimensions, typename elemT>
void 
ImageProcessor<num_dimensions,elemT>::
apply(DiscretisedDensity<num_dimensions,elemT>& density,
		 const DiscretisedDensity<num_dimensions,elemT>& in_density)
  {
    //assert(consistency_check(in_density) == Succeeded::yes);
    if (!filter_built )
      if (build_filter(in_density) == Succeeded::no)
      {
	warning("ImageProcessor::apply: Building filter was unsuccesfull. No filtering done.\n");
	return;
      }
    virtual_apply(density, in_density);
  }

#if 0
template <int num_dimensions, typename elemT>
Succeeded 
ImageProcessor<num_dimensions,elemT>::
consistency_check( const DiscretisedDensity<num_dimensions,elemT>& image) const
{
  return Succeeded::yes;
}
#endif

#if 0
// lines necessary for .cxx, but it's now .inl

// instantiation
template ImageProcessor<3,float>;
#endif

END_NAMESPACE_TOMO
