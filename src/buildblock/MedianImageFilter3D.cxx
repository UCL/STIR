//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Implementations for class MedianImageFilter3D

  \author Sanida Mustafovic
  \author Kris Thielemans

  \date $Date$
  \version $Revision$
*/

#include "tomo/MedianImageFilter3D.h"
#include "CartesianCoordinate3D.h"
#include "DiscretisedDensity.h"


START_NAMESPACE_TOMO

template <typename elemT>
MedianImageFilter3D<elemT>:: MedianImageFilter3D(const CartesianCoordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius.x();
  mask_radius_y = mask_radius.y();
  mask_radius_z = mask_radius.z();
}

template <typename elemT>
MedianImageFilter3D<elemT>:: MedianImageFilter3D()
{
  set_defaults();
}

template <typename elemT>
Succeeded
MedianImageFilter3D<elemT>::virtual_set_up (const DiscretisedDensity<3,elemT>& density)
{

/*   if (consistency_check(density) == Succeeded::no)
      return Succeeded::no;*/
   median_filter = 
     MedianArrayFilter3D<elemT>(Coordinate3D<int>
     (mask_radius_z, mask_radius_y, mask_radius_x));

   return Succeeded::yes;
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  //assert(consistency_check(density) == Succeeded::yes);
  median_filter(density);   
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density, const DiscretisedDensity<3, elemT>& in_density) const
{
  //assert(consistency_check(in_density) == Succeeded::yes);
  median_filter(out_density,in_density);   
}

template <typename elemT>
void
MedianImageFilter3D<elemT>::set_defaults()
{
  mask_radius_x = 0;
  mask_radius_y = 0;
  mask_radius_z = 0;
}

template <typename elemT>
void 
MedianImageFilter3D<elemT>::initialise_keymap()
{
  parser.add_start_key("Median Filter Parameters");
  parser.add_key("mask radius x", &mask_radius_x);
  parser.add_key("mask radius y", &mask_radius_y);
  parser.add_key("mask radius z", &mask_radius_z);
  parser.add_stop_key("END Median Filter Parameters");
}


const char * const 
MedianImageFilter3D<float>::registered_name =
  "Median";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


// Register this class in the ImageProcessor registry
//static MedianImageFilter3D<float>::RegisterIt dummy;

template MedianImageFilter3D<float>;

END_NAMESPACE_TOMO
