#if 0
//
// %W%: %E%
//
/*!

  \file

  \brief 

  \author Sanida Mustafovic
  \author Kris Thielemans

  \date %E%
  \version %I%
*/

#include "tomo/SVLCImageFilter.h"
#include "CartesianCoordinate3D.h"
#include "DiscretisedDensity.h"

// TODO REMOVE
#include <iostream>
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

template <typename elemT>
SVLCImageFilter<elemT>:: SVLCImageFilter(const CartesianCoordinate3D<int>& mask_radius)
{
  mask_radius_x = mask_radius.x();
  mask_radius_y = mask_radius.y();
  mask_radius_z = mask_radius.z();
}

template <typename elemT>
SVLCImageFilter<elemT>:: SVLCImageFilter()
{
  set_defaults();
}

template <typename elemT>
Succeeded
SVLCImageFilter<elemT>::virtual_build_filter (const DiscretisedDensity<3,elemT>& density)
{

   if (consistency_check(density) == Succeeded::no)
      return Succeeded::no;
   svlc_filter = 
     SVLCArrayFilter<elemT>(Coordinate3D<int>
     (mask_radius_z, mask_radius_y, mask_radius_x));

   return Succeeded::yes;
}

template <typename elemT>
void
SVLCImageFilter<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& density) const
{
  assert(consistency_check(density) == Succeeded::yes);
  svlc_filter(density);
   
}

template <typename elemT>
void
SVLCImageFilter<elemT>::virtual_apply(DiscretisedDensity<3, elemT>& out_density, const DiscretisedDensity<3, elemT>& in_density) const
{
  assert(consistency_check(in_density) == Succeeded::yes);
  svlc_filter(out_density,in_density);
   
}

template <typename elemT>
void
SVLCImageFilter<elemT>::set_defaults()
{
  mask_radius_x = 0;
  mask_radius_y = 0;
  mask_radius_z = 0;
}

template <typename elemT>
void 
SVLCImageFilter<elemT>::initialise_keymap()
{
  parser.add_start_key("SVLC Filter Parameters");
  parser.add_key("mask radius x", &mask_radius_x);
  parser.add_key("mask radius y", &mask_radius_y);
  parser.add_key("mask radius z", &mask_radius_z);
  parser.add_stop_key("END SVLC Filter Parameters");
}


const char * const 
SVLCImageFilter<float>::registered_name =
  "DAV";

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

#ifndef TOMO_STATIC
// Register this class in the ImageFilter registry
// At the same time, the compiler will instantiate SVLCImageFilter<float>
static SVLCImageFilter<float>::RegisterIt dummy;

template SVLCImageFilter<float>;
#else


template SVLCImageFilter<float>;
#endif

END_NAMESPACE_TOMO

#endif